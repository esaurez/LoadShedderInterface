#include "comm_agent.h"

CommAgent::CommAgent(const std::string &serverUrl, std::shared_ptr<zmq::context_t> ctxPtr) {
    ctx = ctxPtr;
    if (ctx == nullptr) {
        ctx = std::make_shared<zmq::context_t>(1);
    }
    sock = std::make_unique<zmq::socket_t>(zmq::socket_t(*ctx, ZMQ_REQ));
    sock->connect(serverUrl.c_str());
}

CommAgent::~CommAgent() {
    sock->close();
}

std::unordered_map<unsigned int, double> CommAgent::getUtilityThreshold(const std::unordered_map<unsigned int, float>& perVideoDropRatio,
                                                               const std::string& mode)
{
  ::capnp::MallocMessageBuilder message;
  UtilityMessage::Builder utilMessage = message.initRoot<UtilityMessage>();
  utilMessage.setMessageType(UtilityMessage::Type::UTILITY_THRESHOLD_REQUEST);
  UtilityThresholdRequest::Builder utilThresholdReq = utilMessage.initUtilityThresholdRequest();
  auto dropRequests = utilThresholdReq.initDropRatios(static_cast<unsigned int>(perVideoDropRatio.size()));
  unsigned int idx = 0;
  for (const auto& videoRequest : perVideoDropRatio)
  {
    dropRequests[idx].setVideoIdx(videoRequest.first);
    dropRequests[idx].setDropRatio(videoRequest.second);
    idx++;
  }
  auto wordArray1 = ::capnp::messageToFlatArray(message);
  auto charArray1 = wordArray1.asChars();
  std::shared_ptr<const std::string> serializedOut =
    std::make_shared<const std::string>(charArray1.begin(), charArray1.end());
  zmq::message_t request((void*)serializedOut->data(), serializedOut->size(), NULL);
  // Handling of the reply
  zmq::message_t reply;
  {
    std::lock_guard<std::mutex> lock(agentLock);
    sock->send(std::move(request), zmq::send_flags::none);
    sock->recv(reply);
  }
  std::shared_ptr<const std::string> serialized =
    std::make_shared<const std::string>(static_cast<char*>(reply.data()), reply.size());

  auto num_words = serialized->size() / sizeof(capnp::word);
  auto wordArray = kj::ArrayPtr<capnp::word const>(reinterpret_cast<capnp::word const*>(serialized->data()), num_words);
  ::capnp::FlatArrayMessageReader reader(wordArray);
  auto capnpMessage = reader.getRoot<UtilityMessage>();

  assert(capnpMessage.getMessageType() == UtilityMessage::Type::UTILITY_THRESHOLD_RESPONSE);
  std::unordered_map<unsigned int, double> response;
  const auto& thresholds = capnpMessage.getUtilityThresholdResponse().getThresholds();
  for(const auto& threshold : thresholds)
  {
    response[threshold.getVideoIdx()] = threshold.getThreshold();
  }
  return response;
}

double CommAgent::getUtilityValue(Features::Builder &features, unsigned int videoIdx, const std::string& mode) {
    ::capnp::MallocMessageBuilder message;
    UtilityMessage::Builder utilMessage = message.initRoot<UtilityMessage>();
    utilMessage.setMessageType(UtilityMessage::Type::UTILITY_REQUEST);
    UtilityRequest::Builder utilThresholdReq = utilMessage.initUtilityRequest();
    utilThresholdReq.setFeats(features.asReader());
    utilThresholdReq.setMode(mode);
    utilThresholdReq.setVideoIdx(videoIdx);

    auto wordArray1 = ::capnp::messageToFlatArray(message);
    auto charArray1 = wordArray1.asChars();
    std::shared_ptr<const std::string> serializedOut = std::make_shared<const std::string> (charArray1.begin(), charArray1.end());
    zmq::message_t request((void*)serializedOut->data(), serializedOut->size(), NULL);
    // Handling of the reply
    zmq::message_t reply;
    {
        std::lock_guard<std::mutex> lock(agentLock);
        sock->send (std::move(request), zmq::send_flags::none);
        sock->recv (reply);
    }
    std::shared_ptr<const std::string> serialized = std::make_shared<const std::string>(static_cast<char*>(reply.data()), reply.size());

    auto num_words = serialized->size() / sizeof(capnp::word);
    auto wordArray = kj::ArrayPtr<capnp::word const>(reinterpret_cast<capnp::word const*>(serialized->data()), num_words);
    ::capnp::FlatArrayMessageReader reader(wordArray);
    auto capnpMessage = reader.getRoot<UtilityMessage>();

    assert(capnpMessage.getMessageType() == UtilityMessage::Type::UTILITY_RESPONSE);
    return capnpMessage.getUtilityResponse().getUtility();
}
