#include "sessionmgmt.h"

SessionManagement::SessionManagement(
  size_t max_sessions_
): max_sessions(max_sessions_) {}


SessionManagement::~SessionManagement() {
  clear_all();
}

void SessionManagement::put(
  const std::string session_id_,
  const std::string node_id_,
  std::shared_ptr<GeneralMHAModel> model_inst_
) {
  if (sessions.size() >= max_sessions) {
    auto time_now = std::chrono::system_clock::now();
    auto oldest_session = sessions.begin();
    for (auto it = sessions.begin(); it != sessions.end(); ++it) {
      if (it->second.created_at < oldest_session->second.created_at) {
        oldest_session = it;
      }
    }
    std::cout << "Max sessions reached, removing oldest session: "
              << oldest_session->first << std::endl;
    sessions.erase(oldest_session);
  }

  if (sessions.find(session_id_) != sessions.end()) {
    std::cerr << "Session ID already exists, updating existing session." << std::endl;

    sessions[session_id_] = Session{
      session_id_,
      node_id_,
      std::chrono::system_clock::now(),
      model_inst_
    };
  } else {
    Session new_session{
      session_id_,
      node_id_,
      std::chrono::system_clock::now(),
      model_inst_
    };

    sessions[session_id_] = new_session;
    std::cout << "New session added: " << session_id_ << std::endl;
  }
}

std::shared_ptr<GeneralMHAModel> SessionManagement::get_model(
  const std::string session_id_,
  const std::string node_id_
) {
  auto it = sessions.find(session_id_);
  if (it != sessions.end() && it->second.node_id == node_id_) {
    return it->second.model_instance;
  }
  return nullptr; 
}

std::string SessionManagement::current_sessions() const {
  std::string session_list;
  for (const auto& session : sessions) {
    session_list += "["+session.first+" @ "+session.second.node_id+"] | "+
      session.second.model_instance->shard.model_id+"\n";
  }
  return session_list.empty() ? "No active sessions." : session_list;
}

void SessionManagement::clear_all() {
  sessions.clear();
  std::cout << "All sessions cleared." << std::endl;
}

bool SessionManagement::has_session(
  const std::string session_id_,
  const std::string node_id_
) {
  auto it = sessions.find(session_id_);
  return it != sessions.end() && it->second.node_id == node_id_;
}