#include <iostream>
#include "comm_agent.h"

int main() {
    CommAgent agent("tcp://localhost:5556");
    std::cout << agent.getUtilityThreshold(0) << std::endl; 

    ::capnp::MallocMessageBuilder message;
    Features::Builder features = message.initRoot<Features>();

    std::cout << agent.getUtilityValue(features) << std::endl;

}
