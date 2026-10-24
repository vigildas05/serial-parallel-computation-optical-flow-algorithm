#ifndef SYSINFO_HPP
#define SYSINFO_HPP

#include <string>

// Host description printed alongside benchmark results for reproducibility.
namespace sysinfo {

std::string cpuModel();   // CPUID brand string, or "unknown"
int physicalCores();      // 0 if unknown
int logicalCores();       // hardware threads

} // namespace sysinfo

#endif
