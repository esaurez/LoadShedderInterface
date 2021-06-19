import zmq
import time
import sys
import os
import capnp
import argparse

script_dir=os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../"))

from capnp_serial import messages_capnp
from capnp_serial import mapping_capnp

## Stuttgart folks implement this function ##
def compute_util_threshold(drop_ratio):
    return 0.5 # TODO

## Stuttgart folks implement this function ##
def compute_utility(features):
    return 0.6 # TODO

def handle_util_threshold_req(util_threshold_req):
    drop_ratio = util_threshold_req.dropRatio
    
    # Function call to compute the util threshold for given drop ratio
    util_threshold = compute_util_threshold(drop_ratio)

    reply = messages_capnp.UtilityMessage.new_message()
    reply.messageType = "utilityThresholdResponse"
    reply.init("utilityThresholdResponse")
    reply.utilityThresholdResponse.threshold = util_threshold
    return reply

def handle_util_req(util_req):
    features = util_req.feats.feats

    # Function call to compute the utility for given features
    utility = compute_utility(features)

    reply = messages_capnp.UtilityMessage.new_message()
    reply.messageType = "utilityResponse"
    reply.init("utilityResponse")
    reply.utilityResponse.utility = utility
    return reply

def handle_msg(message):
    util_msg  = messages_capnp.UtilityMessage.from_bytes(message)
    if util_msg.messageType == "utilityThresholdRequest":
        util_threshold_req = util_msg.utilityThresholdRequest
        reply = handle_util_threshold_req(util_threshold_req)
    elif util_msg.messageType == "utilityRequest":
        util_req = util_msg.utilityRequest
        reply = handle_util_req(util_req)
    else:
        raise Exception("Unknown message type received")
    return reply

def start_recving_reqs(socket) :
    while True:
        #  Wait for next request from client
        message = socket.recv()
        reply = handle_msg(message)
        socket.send(reply.to_bytes())

def main(port):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % port)

    start_recving_reqs(socket)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", dest="port", help="Port to listen on", type=int, default=5556)
    args = parser.parse_args()

    main(args.port)
