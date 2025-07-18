#include "helpers.h"

bool Helpers::is_low_memory(
  const std::string os = "linux",
  size_t threshold_mb = 20000
) {

  if (os == "linux") {
    struct sysinfo info;
    if (sysinfo(&info) != 0)
    return false;

    unsigned long total_ram_mb = info.totalram * info.mem_unit / (1024 * 1024);
    return total_ram_mb < threshold_mb;
  }

  return false;
}

bool Helpers::llama_detect(std::string model_name_) {
  static const std::regex rgx_llama(
    "llama",
    std::regex_constants::icase
  );
  
  return std::regex_search(model_name_, rgx_llama);
}

bool Helpers::model_ver_detect(
  const std::string model_name_,
  const std::string target_version_
) {
  static const std::regex rgx_ver(
    target_version_,
    std::regex_constants::icase
  );

  return std::regex_search(model_name_, rgx_ver);
}