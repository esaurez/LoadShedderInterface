#ifndef _COMM_AGENT_H
#define _COMM_AGENT_H

#include <iostream>
#include <zmq.hpp>
#include <capnp/message.h>
#include <capnp/serialize-packed.h>
#include <memory>
#include <mutex>
#include <unordered_map>
#include "mapping.capnp.h"
#include "messages.capnp.h"

class AbstractCommAgent {
public:
    virtual ~AbstractCommAgent() = default;
    virtual std::unordered_map<int, double> getUtilityThreshold(const std::unordered_map<int, float>& perVideoDropRatio, const std::string& mode = "max_cdf") = 0;
    virtual double getUtilityValue(Features::Builder &utilityRequest, int videoIdx, const std::string& mode = "max_cdf") = 0;
};

class CommAgent : public AbstractCommAgent{
public:
    CommAgent(const std::string &serverUrl, std::shared_ptr<zmq::context_t> ctxPtr=nullptr);
    ~CommAgent();
    virtual std::unordered_map<int, double> getUtilityThreshold(const std::unordered_map<int, float>& perVideoDropRatio, const std::string& mode = "max_cdf") override;
    virtual double getUtilityValue(Features::Builder &utilityRequest, int videoIdx, const std::string& mode = "max_cdf") override;
private:
    std::mutex agentLock;
    std::shared_ptr<zmq::context_t> ctx;
    std::unique_ptr<zmq::socket_t> sock;
};


#endif // _COMM_AGENT_H
