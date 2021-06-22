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

class InterfaceReqHandler:
    def __init__(self, port):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:%s" % port)
    
    def run(self):
        while True:
            #  Wait for next request from client
            message = self.socket.recv()
            util_msg  = messages_capnp.UtilityMessage.from_bytes(message)
            reply = self.handle_msg(util_msg)
            self.socket.send(reply.to_bytes())

    def handle_msg(self, util_msg):
        if util_msg.messageType == "utilityThresholdRequest":
            util_threshold_req = util_msg.utilityThresholdRequest
            reply = self.handle_util_threshold_req(util_threshold_req)
        elif util_msg.messageType == "utilityRequest":
            util_req = util_msg.utilityRequest
            reply = self.handle_util_req(util_req)
        else:
            raise Exception("Unknown message type received")
        return reply

    ## Stuttgart folks implement this function ##
    def compute_util_threshold(self, drop_ratio):
        return 0.5 # TODO
    
    ## Stuttgart folks implement this function ##
    def compute_utility(self, features):
        return 0.6 # TODO
    
    def handle_util_threshold_req(self, util_threshold_req):
        drop_ratio = util_threshold_req.dropRatio
        
        # Function call to compute the util threshold for given drop ratio
        util_threshold = self.compute_util_threshold(drop_ratio)
    
        reply = messages_capnp.UtilityMessage.new_message()
        reply.messageType = "utilityThresholdResponse"
        reply.init("utilityThresholdResponse")
        reply.utilityThresholdResponse.threshold = util_threshold
        return reply
    
    def handle_util_req(self, util_req):
        features = util_req.feats.feats
    
        # Function call to compute the utility for given features
        utility = self.compute_utility(features)
    
        reply = messages_capnp.UtilityMessage.new_message()
        reply.messageType = "utilityResponse"
        reply.init("utilityResponse")
        reply.utilityResponse.utility = utility
        return reply

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", dest="port", help="Port to listen on", type=int, default=5556)
    args = parser.parse_args()

    interface = InterfaceReqHandler(args.port)
    interface.run()
