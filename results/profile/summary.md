Nsight Compute, 1920x1080, iteration kernel, median over sampled launches (GPU at its normal clocks).

| version | duration (µs) | achieved occupancy | theoretical occupancy | DRAM throughput (% peak) | DRAM GB/s | L1 throughput (% peak) | L2 throughput (% peak) | memory SOL (% peak) | SM SOL (% peak) | DRAM bytes / pixel | L1 hit | L2 hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cuda-v1 | 616.9 | 90.1% | 100.0% | 96.0% | 107.3 | 19.6% | 52.4% | 96.0% | 17.1% | 32.0 | 37.1% | 36.2% |
| cuda-v2 | 617.9 | 92.1% | 100.0% | 95.9% | 107.2 | 31.2% | 42.8% | 95.9% | 31.2% | 32.0 | 0.8% | 36.0% |
| cuda-v3 | 546.4 | 87.1% | 100.0% | 95.3% | 106.5 | 18.2% | 51.4% | 95.3% | 12.7% | 28.1 | 41.1% | 36.4% |
| cuda-rb | 634.7 | 86.4% | 100.0% | 94.7% | 105.9 | 14.9% | 38.3% | 94.7% | 5.3% | 32.5 | 56.5% | 26.9% |

Sampled launches per kernel: cuda-v1: 10, cuda-v2: 10, cuda-v3: 10, cuda-rb: 10
