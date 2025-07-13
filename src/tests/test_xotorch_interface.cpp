#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <netinet/in.h>

#include "./../general_mha_model.h"
using json = nlohmann::json;

