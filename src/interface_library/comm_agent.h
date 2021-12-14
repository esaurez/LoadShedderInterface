#ifndef _COMM_AGENT_H
#define _COMM_AGENT_H

#include <iostream>
#include <zmq.hpp>
#include <capnp/message.h>
#include <capnp/serialize-packed.h>
#include <memory>
#include <mutex>
#include "mapping.capnp.h"
#include "messages.capnp.h"

class AbstractCommAgent {
public:
    virtual ~AbstractCommAgent() = default;
    virtual double getUtilityThreshold(float dropRatio, const std::string& mode = "max_cdf") = 0;
    virtual double getUtilityValue(Features::Builder &utilityRequest, const std::string& mode = "max_cdf") = 0;
};

class CommAgent : public AbstractCommAgent{
public:
    CommAgent(const std::string &serverUrl, std::shared_ptr<zmq::context_t> ctxPtr=nullptr);
    ~CommAgent();
    virtual double getUtilityThreshold(float dropRatio, const std::string& mode = "max_cdf") override;
    virtual double getUtilityValue(Features::Builder &utilityRequest, const std::string& mode = "max_cdf") override;
private:
    std::mutex agentLock;
    std::shared_ptr<zmq::context_t> ctx;
    std::unique_ptr<zmq::socket_t> sock;
};


#endif // _COMM_AGENT_H
