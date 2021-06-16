#include "comm_agent.h"

CommAgent::CommAgent(const std::string &serverUrl) : context(zmq::context_t(1)), sock(zmq::socket_t(context, ZMQ_REQ)) {
    sock.connect(serverUrl.c_str());
}

CommAgent::~CommAgent() {
    sock.close();
}

float CommAgent::getUtilityThreshold() {
    ::capnp::MallocMessageBuilder message;
    Point::Builder pointBuilder = message.initRoot<Point>();
    pointBuilder.setX(100);
    auto wordArray1 = ::capnp::messageToFlatArray(message);
    auto charArray1 = wordArray1.asChars();
    std::shared_ptr<const std::string> serializedOut = std::make_shared<const std::string> (charArray1.begin(), charArray1.end());
    zmq::message_t request((void*)serializedOut->data(), serializedOut->size(), NULL);
    sock.send (request, 0);

    // Handling of the reply
    zmq::message_t reply;
    sock.recv ( &reply, 0);
    std::shared_ptr<const std::string> serialized = std::make_shared<const std::string>(static_cast<char*>(reply.data()), reply.size());

    auto num_words = serialized->size() / sizeof(capnp::word);
    auto wordArray = kj::ArrayPtr<capnp::word const>(reinterpret_cast<capnp::word const*>(serialized->data()), num_words);
    ::capnp::FlatArrayMessageReader reader(wordArray);
    auto capnpMessage = reader.getRoot<Point>();

    std::cout << "Received message " << capnpMessage.getX() << std::endl;
    return 0;
}

float CommAgent::getUtilityValue() {
    return getUtilityThreshold();    
}
