// Session Managment
#ifndef SESSIONMGMT_H
#define SESSIONMGMT_H

#include <unordered_map>
#include <memory>
#include <string>
#include <chrono>
#include <iostream>
#include <limits>

#include "../general_mha_model.h"

class SessionManagement {
  public:
    SessionManagement(size_t max_sessions_=100);
    ~SessionManagement();
    void put(
      const std::string session_id_,
      const std::string node_id_,
      std::shared_ptr<GeneralMHAModel> model_inst_
    );
    std::shared_ptr<GeneralMHAModel> get_model(
      const std::string session_id_,
      const std::string node_id_
    );
    std::string current_sessions() const;
    void clear_all();
    bool has_session(
      const std::string session_id_,
      const std::string node_id_
    );

    struct Session {
      std::string session_id;
      std::string node_id;
      std::chrono::system_clock::time_point created_at;
      std::shared_ptr<GeneralMHAModel> model_instance;
    };

    size_t max_sessions;
    std::unordered_map<std::string, Session> sessions;
};

#endif
