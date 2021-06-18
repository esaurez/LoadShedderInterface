import zmq
import time
import sys
import os
import capnp

script_dir=os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../"))

from capnp_serial import messages_capnp
from capnp_serial import mapping_capnp

port = "5556"
if len(sys.argv) > 1:
    port =  sys.argv[1]

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:%s" % port)

while True:
    #  Wait for next request from client
    message = socket.recv()
    util_msg  = messages_capnp.UtilityMessage.from_bytes(message)

    if util_msg.messageType == "utilityThresholdRequest":
        reply = messages_capnp.UtilityMessage.new_message()
        reply.messageType = "utilityThresholdResponse"
        reply.init("utilityThresholdResponse")
        reply.utilityThresholdResponse.threshold = 75
        socket.send(reply.to_bytes())
    elif util_msg.messageType == "utilityRequest":
        reply = messages_capnp.UtilityMessage.new_message()
        reply.messageType = "utilityResponse"
        reply.init("utilityResponse")
        reply.utilityResponse.utility = 0.6
        socket.send(reply.to_bytes())
