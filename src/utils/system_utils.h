// System utilities
#ifndef SYSTEM_UTILS_H
#define SYSTEM_UTILS_H

#include <iostream>
#include <string>
#include <sys/sysinfo.h>

bool is_low_memory(
    const std::string os = "linux",
    size_t threshold_mb = 20000) {
    
    if(os == "linux") {
        struct sysinfo info;
        if (sysinfo(&info) != 0) return false;

        unsigned long total_ram_mb = info.totalram * info.mem_unit / (1024 * 1024);
        return total_ram_mb < threshold_mb;
    }
    
    return false;
}

#endif