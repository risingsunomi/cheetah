// Helper functions
#ifndef HELPERS_H
#define HELPERS_H

#include <iostream>
#include <string>
#include <sys/sysinfo.h>
#include <regex>

class Helpers {
  public:
  bool is_low_memory(const std::string os_, size_t threshold_mb_);
  bool llama_detect(std::string model_name_);
  bool model_ver_detect(
    const std::string model_name_,
    const std::string target_version_
  );

  const std::string os = "linux";
  size_t threshold_mb = 20000;
};

#endif