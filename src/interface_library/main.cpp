#include <iostream>
#include "comm_agent.h"

int main() {
    std::string mode = "max_cdf";
    CommAgent agent("tcp://localhost:5556");
    std::cout << agent.getUtilityThreshold(0, mode) << std::endl; 

    ::capnp::MallocMessageBuilder message;
    Features::Builder features = message.initRoot<Features>();

    std::cout << agent.getUtilityValue(features, mode) << std::endl;

}
