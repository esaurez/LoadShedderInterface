#include <iostream>
#include "comm_agent.h"

int main() {
    CommAgent agent("tcp://localhost:5556");
    agent.getUtilityThreshold();
}
