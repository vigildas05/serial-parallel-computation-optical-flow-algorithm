#include "opt_flow_serial.hpp"
#include "opt_flow_parallel.hpp"
#include "flow_viz.hpp"
#include "hs_common.hpp"
#include "sysinfo.hpp"
#ifdef OFP_HAVE_CUDA
#include "opt_flow_cuda.hpp"
#endif

#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

namespace {

const char *kUsage = R"(Horn-Schunck optical flow: serial, OpenMP and CUDA implementations.

Usage:
  opt-flow-parallel <frame1> <frame2>            original demo: serial + OpenMP, arrow plots, GUI
  opt-flow-parallel <command> [frame1 frame2] [options]

Commands:
  bench        time every (impl, size, iterations, threads) combination
  validate     compare Jacobi variants against OpenMP Jacobi, cuda-rb against serial-rb
  convergence  record the fixed-point residual after every iteration
  flow         save HSV color-coded flow images for visual comparison

Options (lists are comma separated):
  --impl LIST       serial-gs, serial-jacobi, serial-rb, omp, omp-opt, cuda-v1, cuda-v2, cuda-v3, cuda-rb, all
  --size LIST       WxH or "native" (input frames are resized)     [native]
  --iters LIST      iteration counts                                [100]
  --threads LIST    OpenMP thread counts, "max" = omp_get_max_threads()  [max]
  --alpha A         regularization weight                           [0.5]
  --runs N          timed runs per configuration (median reported)  [10]
  --warmup N        untimed warm-up runs per configuration          [2]
  --csv FILE        write results as CSV (bench, convergence)
  --append          append to --csv instead of overwriting
  --out DIR         output directory for 'flow'                     [output]
Frames default to data/rm1.jpg data/rm2.jpg.
)";

// ---------------------------------------------------------------------------
// Implementations
// ---------------------------------------------------------------------------
enum class Impl { SerialGS, SerialJacobi, SerialRB, OMP, OMPOpt, CudaV1, CudaV2, CudaV3, CudaRB };

struct ImplInfo {
    Impl id;
    const char *name;
    bool gpu;
};

const ImplInfo kImpls[] = {
    {Impl::SerialGS, "serial-gs", false},  {Impl::SerialJacobi, "serial-jacobi", false},
    {Impl::SerialRB, "serial-rb", false},
    {Impl::OMP, "omp", false},             {Impl::OMPOpt, "omp-opt", false},
    {Impl::CudaV1, "cuda-v1", true},
    {Impl::CudaV2, "cuda-v2", true},       {Impl::CudaV3, "cuda-v3", true},
    {Impl::CudaRB, "cuda-rb", true},
};

const ImplInfo &info(Impl id)
{
    for (const auto &i : kImpls)
        if (i.id == id)
            return i;
    throw logic_error("unknown impl");
}

bool cudaAvailable()
{
#ifdef OFP_HAVE_CUDA
    return true;
#else
    return false;
#endif
}

// All timings in milliseconds; transfers are NaN for CPU implementations.
struct Timing {
    double h2d = NAN, compute = 0, d2h = NAN, total = 0;
};

// Runs any implementation; keeps the CUDA solver (device buffers) alive
// across runs of the same image size so allocation is never timed.
class Runner {
public:
    Timing run(Impl impl, const Mat &I1, const Mat &I2, Mat &u, Mat &v,
               float alpha, int iters, int threads,
               const hs::IterationCallback &cb = {})
    {
        if (info(impl).gpu)
            return runGpu(impl, I1, I2, u, v, alpha, iters, cb);

        // OpenCV's own threading (used inside GaussianBlur/Sobel) follows the
        // same thread count, so "serial" really is single-threaded.
        const int prevOmp = omp_get_max_threads(), prevCv = cv::getNumThreads();
        omp_set_num_threads(threads);
        cv::setNumThreads(threads);

        Timing t;
        const auto start = chrono::steady_clock::now();
        switch (impl) {
        case Impl::SerialGS: hornSchunckSerial(I1, I2, u, v, alpha, iters, cb); break;
        case Impl::SerialJacobi: hornSchunckSerialJacobi(I1, I2, u, v, alpha, iters, cb); break;
        case Impl::SerialRB: hornSchunckSerialRedBlack(I1, I2, u, v, alpha, iters, cb); break;
        case Impl::OMP: hornSchunckParallel(I1, I2, u, v, alpha, iters, cb); break;
        case Impl::OMPOpt: hornSchunckParallelOpt(I1, I2, u, v, alpha, iters, cb); break;
        default: break;
        }
        t.total = t.compute = chrono::duration<double, milli>(chrono::steady_clock::now() - start).count();

        omp_set_num_threads(prevOmp);
        cv::setNumThreads(prevCv);
        return t;
    }

private:
#ifdef OFP_HAVE_CUDA
    unique_ptr<hs::gpu::Solver> solver_;
    Size solverSize_;

    Timing runGpu(Impl impl, const Mat &I1, const Mat &I2, Mat &u, Mat &v,
                  float alpha, int iters, const hs::IterationCallback &cb)
    {
        if (!solver_ || solverSize_ != I1.size()) {
            solver_.reset(); // free the old buffers first
            solver_ = make_unique<hs::gpu::Solver>(I1.rows, I1.cols);
            solverSize_ = I1.size();
        }
        const hs::gpu::Kernel k = impl == Impl::CudaV1   ? hs::gpu::Kernel::Naive
                                  : impl == Impl::CudaV2 ? hs::gpu::Kernel::Shared
                                  : impl == Impl::CudaV3 ? hs::gpu::Kernel::Fused
                                                         : hs::gpu::Kernel::RedBlack;
        hs::gpu::Timings gt;
        solver_->run(I1, I2, u, v, alpha, iters, k, &gt, cb);
        return {gt.h2d_ms, gt.compute_ms, gt.d2h_ms, gt.total_ms};
    }
#else
    Timing runGpu(Impl, const Mat &, const Mat &, Mat &, Mat &, float, int,
                  const hs::IterationCallback &)
    {
        throw runtime_error("this build has no CUDA support");
    }
#endif
};

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------
struct Options {
    string command;
    string frame1 = "data/rm1.jpg", frame2 = "data/rm2.jpg";
    vector<Impl> impls;
    vector<Size> sizes{Size()}; // empty Size = native
    vector<int> iters{100};
    vector<int> threads;
    float alpha = 0.5f;
    int runs = 10, warmup = 2;
    string csv;
    bool append = false;
    string outDir = "output";
};

vector<string> splitList(const string &s)
{
    vector<string> out;
    stringstream ss(s);
    string item;
    while (getline(ss, item, ','))
        if (!item.empty())
            out.push_back(item);
    return out;
}

vector<Impl> parseImpls(const string &s)
{
    vector<Impl> out;
    for (const string &name : splitList(s)) {
        bool found = false;
        for (const auto &i : kImpls) {
            if (name == "all" || name == i.name) {
                if (i.gpu && !cudaAvailable()) {
                    if (name != "all")
                        throw runtime_error(name + " requested but this build has no CUDA support");
                    continue;
                }
                if (find(out.begin(), out.end(), i.id) == out.end())
                    out.push_back(i.id);
                found = true;
            }
        }
        if (!found && name != "all")
            throw runtime_error("unknown implementation: " + name);
    }
    return out;
}

Options parseArgs(int argc, char **argv)
{
    Options o;
    o.command = argv[1];
    vector<string> positional;
    for (int i = 2; i < argc; ++i) {
        const string a = argv[i];
        auto next = [&]() -> string {
            if (i + 1 >= argc)
                throw runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "--impl") o.impls = parseImpls(next());
        else if (a == "--size") {
            o.sizes.clear();
            for (const string &s : splitList(next())) {
                if (s == "native") {
                    o.sizes.push_back(Size());
                    continue;
                }
                const size_t x = s.find('x');
                const int w = x == string::npos ? 0 : atoi(s.substr(0, x).c_str());
                const int h = x == string::npos ? 0 : atoi(s.substr(x + 1).c_str());
                if (w <= 2 || h <= 2)
                    throw runtime_error("bad size: " + s);
                o.sizes.emplace_back(w, h);
            }
        }
        else if (a == "--iters") {
            o.iters.clear();
            for (const string &s : splitList(next())) o.iters.push_back(stoi(s));
        }
        else if (a == "--threads") {
            for (const string &s : splitList(next()))
                o.threads.push_back(s == "max" ? omp_get_max_threads() : stoi(s));
        }
        else if (a == "--alpha") o.alpha = stof(next());
        else if (a == "--runs") o.runs = stoi(next());
        else if (a == "--warmup") o.warmup = stoi(next());
        else if (a == "--csv") o.csv = next();
        else if (a == "--append") o.append = true;
        else if (a == "--out") o.outDir = next();
        else if (a.rfind("--", 0) == 0) throw runtime_error("unknown option: " + a);
        else positional.push_back(a);
    }
    if (positional.size() == 2) {
        o.frame1 = positional[0];
        o.frame2 = positional[1];
    } else if (!positional.empty()) {
        throw runtime_error("expected exactly two frame paths");
    }
    if (o.threads.empty())
        o.threads.push_back(omp_get_max_threads());
    if (o.runs < 1 || o.warmup < 0)
        throw runtime_error("--runs must be >= 1 and --warmup >= 0");
    return o;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
Mat loadFrame(const string &path)
{
    Mat I = imread(path, IMREAD_GRAYSCALE);
    if (I.empty())
        throw runtime_error("cannot load image: " + path);
    I.convertTo(I, CV_32F, 1.0 / 255.0);
    return I;
}

// Resize both frames to `size` (no-op for an empty size).
void prepareFrames(const Mat &I1, const Mat &I2, Size size, Mat &A, Mat &B)
{
    if (size.empty() || size == I1.size()) {
        A = I1;
        B = I2;
        return;
    }
    const bool shrink = size.width <= I1.cols && size.height <= I1.rows;
    const int interp = shrink ? INTER_AREA : INTER_LINEAR;
    resize(I1, A, size, 0, 0, interp);
    resize(I2, B, size, 0, 0, interp);
}

double median(vector<double> v)
{
    if (v.empty())
        return NAN;
    sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

string sizeStr(Size s)
{
    return to_string(s.width) + "x" + to_string(s.height);
}

string systemReport()
{
    ostringstream os;
    os << "CPU:    " << sysinfo::cpuModel() << " (" << sysinfo::physicalCores() << " cores, "
       << sysinfo::logicalCores() << " threads)\n"
       << "OpenMP: max threads " << omp_get_max_threads() << "\n"
       << "OpenCV: " << CV_VERSION << "\n";
#ifdef OFP_HAVE_CUDA
    os << "GPU:    " << hs::gpu::deviceSummary() << "\n";
#else
    os << "GPU:    (built without CUDA)\n";
#endif
    return os.str();
}

ofstream openCsv(const Options &o, const string &header)
{
    const filesystem::path p(o.csv);
    if (p.has_parent_path())
        filesystem::create_directories(p.parent_path());
    const bool writeHeader = !o.append || !filesystem::exists(p) || filesystem::file_size(p) == 0;
    ofstream f(p, o.append ? ios::app : ios::trunc);
    if (!f)
        throw runtime_error("cannot open " + o.csv);
    if (writeHeader)
        f << header << "\n";
    return f;
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------
int cmdBench(const Options &o)
{
    const Mat F1 = loadFrame(o.frame1), F2 = loadFrame(o.frame2);
    const vector<Impl> impls = o.impls.empty() ? parseImpls("all") : o.impls;

    cout << systemReport() << "\n";
    ofstream csv;
    if (!o.csv.empty()) {
        csv = openCsv(o, "impl,width,height,iterations,threads,runs,h2d_ms,compute_ms,d2h_ms,"
                         "total_ms,total_min_ms,total_max_ms");
        ofstream(filesystem::path(o.csv).replace_filename("system_info.txt")) << systemReport();
    }

    cout << left << setw(15) << "impl" << setw(11) << "size" << setw(7) << "iters" << setw(8)
         << "threads" << right << setw(12) << "total[ms]" << setw(12) << "compute" << setw(10)
         << "h2d" << setw(10) << "d2h" << "   (medians of " << o.runs << " runs)\n";

    Runner runner;
    for (Size size : o.sizes) {
        Mat I1, I2;
        prepareFrames(F1, F2, size, I1, I2);
        for (int iters : o.iters) {
            for (Impl impl : impls) {
                vector<int> threadList{1};
                if (impl == Impl::OMP || impl == Impl::OMPOpt) threadList = o.threads;
                for (int threads : threadList) {
                    Mat u, v;
                    for (int w = 0; w < o.warmup; ++w)
                        runner.run(impl, I1, I2, u, v, o.alpha, iters, threads);
                    vector<double> h2d, comp, d2h, total;
                    for (int r = 0; r < o.runs; ++r) {
                        const Timing t = runner.run(impl, I1, I2, u, v, o.alpha, iters, threads);
                        h2d.push_back(t.h2d);
                        comp.push_back(t.compute);
                        d2h.push_back(t.d2h);
                        total.push_back(t.total);
                    }
                    const bool gpu = info(impl).gpu;
                    const double mTotal = median(total), mComp = median(comp);
                    const double mH2d = gpu ? median(h2d) : NAN, mD2h = gpu ? median(d2h) : NAN;
                    const auto [mn, mx] = minmax_element(total.begin(), total.end());

                    cout << left << setw(15) << info(impl).name << setw(11) << sizeStr(I1.size())
                         << setw(7) << iters << setw(8) << (gpu ? string("-") : to_string(threads))
                         << right << fixed << setprecision(3) << setw(12) << mTotal << setw(12)
                         << mComp << setw(10) << mH2d << setw(10) << mD2h << "\n";
                    if (csv) {
                        auto num = [](double x) { return isnan(x) ? string() : to_string(x); };
                        csv << info(impl).name << "," << I1.cols << "," << I1.rows << "," << iters
                            << "," << (gpu ? string() : to_string(threads)) << "," << o.runs << ","
                            << num(mH2d) << "," << num(mComp) << "," << num(mD2h) << ","
                            << num(mTotal) << "," << num(*mn) << "," << num(*mx) << "\n";
                        csv.flush();
                    }
                }
            }
        }
    }
    return 0;
}

// Flow differences between a candidate and a reference solution.
struct FlowError {
    double maxAbsU, maxAbsV, meanEpe;
};

FlowError compareFlow(const Mat &u, const Mat &v, const Mat &uRef, const Mat &vRef)
{
    Mat du = u - uRef, dv = v - vRef, epe;
    magnitude(du, dv, epe);
    return {norm(du, NORM_INF), norm(dv, NORM_INF), mean(epe)[0]};
}

int cmdValidate(const Options &o)
{
    // CPU Jacobi variants perform the same float operations in the same order
    // as the reference and must match it bit for bit.
    // CUDA: float32 vs float32 with different FMA contraction / evaluation
    // order. Flow magnitudes here are up to ~10 px; the naive kernel measured
    // max |d| ~3e-5 and mean EPE ~3e-7 after 1000 iterations, so these
    // leave ~3x headroom while still catching any real indexing/boundary bug.
    const double kMaxAbsTol = 1e-4, kMeanEpeTol = 1e-6;

    const Mat F1 = loadFrame(o.frame1), F2 = loadFrame(o.frame2);
    // Jacobi variants are checked against omp, cuda-rb against serial-rb.
    // serial-gs (lexicographic order) has no counterpart to compare with.
    vector<Impl> impls;
    for (Impl i : (o.impls.empty() ? parseImpls("all") : o.impls))
        if (i != Impl::SerialGS && i != Impl::OMP && i != Impl::SerialRB)
            impls.push_back(i);

    cout << systemReport() << "\n";
    Runner runner;
    bool allPass = true;
    for (Size size : o.sizes) {
        Mat I1, I2;
        prepareFrames(F1, F2, size, I1, I2);
        for (int iters : o.iters) {
            Mat uRef, vRef, uRefRB, vRefRB;
            runner.run(Impl::OMP, I1, I2, uRef, vRef, o.alpha, iters, omp_get_max_threads());
            cout << sizeStr(I1.size()) << ", " << iters << " iterations, alpha " << o.alpha
                 << " (reference omp Jacobi, max |flow| = " << viz::maxMagnitude(uRef, vRef) << " px)\n";
            for (Impl impl : impls) {
                const bool rb = impl == Impl::CudaRB;
                if (rb && uRefRB.empty())
                    runner.run(Impl::SerialRB, I1, I2, uRefRB, vRefRB, o.alpha, iters, 1);
                Mat u, v;
                runner.run(impl, I1, I2, u, v, o.alpha, iters, 1);
                const FlowError e = rb ? compareFlow(u, v, uRefRB, vRefRB) : compareFlow(u, v, uRef, vRef);
                const bool pass = info(impl).gpu
                    ? max(e.maxAbsU, e.maxAbsV) <= kMaxAbsTol && e.meanEpe <= kMeanEpeTol
                    : max(e.maxAbsU, e.maxAbsV) == 0.0;
                allPass &= pass;
                cout << "  " << left << setw(15) << info(impl).name << scientific << setprecision(3)
                     << "max|du| " << e.maxAbsU << "  max|dv| " << e.maxAbsV << "  mean EPE "
                     << e.meanEpe << "  " << (pass ? "PASS" : "FAIL") << (rb ? "  (vs serial-rb)" : "")
                     << defaultfloat << "\n";
            }
        }
    }
    cout << (allPass ? "All checks passed" : "SOME CHECKS FAILED") << " (CPU: bit-exact; CUDA: max abs "
         << kMaxAbsTol << ", mean EPE " << kMeanEpeTol << ")\n";
    return allPass ? 0 : 1;
}

int cmdConvergence(const Options &o)
{
    const Mat F1 = loadFrame(o.frame1), F2 = loadFrame(o.frame2);
    Mat I1, I2;
    prepareFrames(F1, F2, o.sizes.front(), I1, I2);
    const int iters = *max_element(o.iters.begin(), o.iters.end());
    const vector<Impl> impls = o.impls.empty() ? parseImpls("serial-gs,serial-jacobi") : o.impls;

    // Residual is always evaluated on the CPU derivatives so every
    // implementation is measured against the same equations.
    Mat Ix, Iy, It;
    hs::computeDerivatives(I1, I2, Ix, Iy, It);
    const Mat denom = hs::computeDenominator(Ix, Iy, o.alpha);
    const Mat zero = Mat::zeros(I1.size(), CV_32F);
    const double r0 = hs::fixedPointResidual(zero, zero, Ix, Iy, It, denom);

    ofstream csv;
    if (!o.csv.empty())
        csv = openCsv(o, "impl,width,height,iteration,residual");

    Runner runner;
    for (Impl impl : impls) {
        vector<double> res{r0};
        auto cb = [&](int, const Mat &u, const Mat &v) {
            res.push_back(hs::fixedPointResidual(u, v, Ix, Iy, It, denom));
        };
        Mat u, v;
        runner.run(impl, I1, I2, u, v, o.alpha, iters, omp_get_max_threads(), cb);
        cout << left << setw(15) << info(impl).name << " residual: iter 0 " << scientific
             << setprecision(3) << res.front() << ", iter " << iters << " " << res.back()
             << defaultfloat << "\n";
        if (csv)
            for (size_t k = 0; k < res.size(); ++k)
                csv << info(impl).name << "," << I1.cols << "," << I1.rows << "," << k << ","
                    << setprecision(9) << res[k] << "\n";
    }
    return 0;
}

int cmdFlow(const Options &o)
{
    const Mat F1 = loadFrame(o.frame1), F2 = loadFrame(o.frame2);
    Mat I1, I2;
    prepareFrames(F1, F2, o.sizes.front(), I1, I2);
    const int iters = o.iters.front();
    const vector<Impl> impls = o.impls.empty() ? parseImpls("all") : o.impls;
    filesystem::create_directories(o.outDir);

    Runner runner;
    float maxMag = -1.f; // shared color scale, taken from the first implementation
    for (Impl impl : impls) {
        Mat u, v;
        runner.run(impl, I1, I2, u, v, o.alpha, iters, omp_get_max_threads());
        if (maxMag < 0.f)
            maxMag = viz::maxMagnitude(u, v);
        const string path = (filesystem::path(o.outDir) /
                             (string("flow_hsv_") + info(impl).name + ".png")).string();
        imwrite(path, viz::flowToHSV(u, v, maxMag));
        cout << "wrote " << path << "\n";
    }
    cout << "color scale: value = |flow| / " << maxMag << " px, hue = direction ("
         << sizeStr(I1.size()) << ", " << iters << " iterations)\n";
    return 0;
}

// The original demo, unchanged in behaviour.
int legacyDemo(const char *frame1, const char *frame2)
{
    Mat I1 = imread(frame1, IMREAD_GRAYSCALE);
    Mat I2 = imread(frame2, IMREAD_GRAYSCALE);
    if (I1.empty() || I2.empty()) {
        cerr << "Error: Cannot load images." << endl;
        return -1;
    }

    I1.convertTo(I1, CV_32F, 1.0 / 255.0);
    I2.convertTo(I2, CV_32F, 1.0 / 255.0);

    Mat u_serial, v_serial, u_parallel, v_parallel;

    // Serial timing
    auto start_serial = chrono::high_resolution_clock::now();
    hornSchunckSerial(I1, I2, u_serial, v_serial, 0.5f, 100);
    auto end_serial = chrono::high_resolution_clock::now();

    double serial_time = chrono::duration<double>(end_serial - start_serial).count();
    cout << "Serial Horn-Schunck time: " << serial_time << " seconds" << endl;

    // Parallel timing
    auto start_parallel = chrono::high_resolution_clock::now();
    hornSchunckParallel(I1, I2, u_parallel, v_parallel, 0.5f, 100);
    auto end_parallel = chrono::high_resolution_clock::now();

    double parallel_time = chrono::duration<double>(end_parallel - start_parallel).count();
    cout << "Parallel Horn-Schunck time: " << parallel_time << " seconds" << endl;

    Mat flow_serial = viz::flowArrows(u_serial, v_serial);
    Mat flow_parallel = viz::flowArrows(u_parallel, v_parallel);

    imwrite("output/flow_serial.png", flow_serial);
    imwrite("output/flow_parallel.png", flow_parallel);

    imshow("Serial Flow", flow_serial);
    imshow("Parallel Flow", flow_parallel);
    waitKey(0);

    return 0;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 2 || string(argv[1]) == "--help" || string(argv[1]) == "-h") {
        cout << kUsage;
        return argc < 2 ? -1 : 0;
    }

    static const map<string, int (*)(const Options &)> commands = {
        {"bench", cmdBench}, {"validate", cmdValidate},
        {"convergence", cmdConvergence}, {"flow", cmdFlow}};

    const auto cmd = commands.find(argv[1]);
    if (cmd == commands.end()) {
        if (argc == 3)
            return legacyDemo(argv[1], argv[2]);
        cerr << kUsage;
        return -1;
    }

    try {
        return cmd->second(parseArgs(argc, argv));
    } catch (const exception &e) {
        cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
