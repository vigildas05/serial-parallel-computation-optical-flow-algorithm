#include "sysinfo.hpp"

#include <cstring>
#include <thread>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <intrin.h>
#include <vector>
#elif defined(__linux__)
#include <fstream>
#include <set>
#include <utility>
#endif

#if !defined(_WIN32) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#endif

namespace sysinfo {

std::string cpuModel()
{
    unsigned int regs[12] = {};
#if defined(_WIN32)
    int r[4];
    __cpuid(r, 0x80000000);
    if (static_cast<unsigned>(r[0]) < 0x80000004u)
        return "unknown";
    for (int i = 0; i < 3; ++i)
        __cpuid(reinterpret_cast<int *>(regs + 4 * i), 0x80000002 + i);
#elif defined(__x86_64__) || defined(__i386__)
    if (__get_cpuid_max(0x80000000, nullptr) < 0x80000004u)
        return "unknown";
    for (unsigned i = 0; i < 3; ++i)
        __get_cpuid(0x80000002 + i, &regs[4 * i], &regs[4 * i + 1], &regs[4 * i + 2], &regs[4 * i + 3]);
#else
    return "unknown";
#endif
    char brand[49] = {};
    std::memcpy(brand, regs, 48);
    std::string s(brand);
    const auto b = s.find_first_not_of(' ');
    const auto e = s.find_last_not_of(' ');
    return b == std::string::npos ? "unknown" : s.substr(b, e - b + 1);
}

int physicalCores()
{
#if defined(_WIN32)
    DWORD len = 0;
    GetLogicalProcessorInformation(nullptr, &len);
    std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> info(len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    if (!GetLogicalProcessorInformation(info.data(), &len))
        return 0;
    int cores = 0;
    for (const auto &i : info)
        cores += i.Relationship == RelationProcessorCore;
    return cores;
#elif defined(__linux__)
    // Count unique (physical id, core id) pairs.
    std::ifstream f("/proc/cpuinfo");
    std::set<std::pair<int, int>> cores;
    std::string line;
    int phys = 0;
    while (std::getline(f, line)) {
        if (line.rfind("physical id", 0) == 0)
            phys = std::stoi(line.substr(line.find(':') + 1));
        else if (line.rfind("core id", 0) == 0)
            cores.insert({phys, std::stoi(line.substr(line.find(':') + 1))});
    }
    return static_cast<int>(cores.size());
#else
    return 0;
#endif
}

int logicalCores()
{
    return static_cast<int>(std::thread::hardware_concurrency());
}

} // namespace sysinfo
