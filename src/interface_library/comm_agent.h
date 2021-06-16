#ifndef _COMM_AGENT_H
#define _COMM_AGENT_H

#include <iostream>
#include <zmq.hpp>
#include <capnp/message.h>
#include <capnp/serialize-packed.h>
#include <memory>
#include "mapping.capnp.h"

class CommAgent {
public:
    CommAgent(const std::string &serverUrl);
    ~CommAgent();
    float getUtilityThreshold();
    float getUtilityValue();
private:
    zmq::context_t context;
    zmq::socket_t sock;
};


#endif // _COMM_AGENT_H
