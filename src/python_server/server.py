import zmq
import time
import sys
import os
import capnp
import argparse
from configobj import ConfigObj

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../"))

from capnp_serial import mapping_capnp,messages_capnp

import model.build_model
import mapping_features

from model.sv_matrix_model import *

class InterfaceReqHandler:

    def __init__(self, port, properties_file):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:%s" % port)

        self.model = SVMatrixModel(properties_file)

    def run(self):
        while True:
            #  Wait for next request from client
            message = self.socket.recv()
            util_msg = messages_capnp.UtilityMessage.from_bytes(message)
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
    def compute_util_threshold(self, drop_ratio, vid_idx):
        return self.model.get_utility_threshold(drop_ratio, vid_idx)

    ## Stuttgart folks implement this function ##
    def compute_utility(self, features, mode):
        return self.model.get_utility(features)

    def handle_util_threshold_req(self, util_threshold_req):
        mode = util_threshold_req.mode

        reply = messages_capnp.UtilityMessage.new_message()
        reply.messageType = "utilityThresholdResponse"
        reply.init("utilityThresholdResponse")
        thresholds = reply.utilityThresholdResponse.init("thresholds", len(util_threshold_req.dropRatios))

        # Function call to compute the util threshold for given drop ratio
        idx = 0
        for vid_drop_ratio in util_threshold_req.dropRatios:
            vid_idx = vid_drop_ratio.videoIdx
            drop_ratio = vid_drop_ratio.dropRatio
            util_threshold = self.compute_util_threshold(drop_ratio, vid_idx)
            thresholds[idx].videoIdx = vid_idx
            thresholds[idx].threshold = util_threshold
            idx += 1
        return reply

    def handle_util_req(self, util_req):
        #features = util_req.feats.feats
        features = util_req.feats
        mode = util_req.mode

        # Function call to compute the utility for given features
        utility = self.compute_utility(features, mode)

        reply = messages_capnp.UtilityMessage.new_message()
        reply.messageType = "utilityResponse"
        reply.init("utilityResponse")
        reply.utilityResponse.utility = utility
        return reply

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", dest="port", help="Port to listen on", type=int, default=5556)
    parser.add_argument("--pf", "--properties_file", dest="properties_file", required=True, type=str,
                        help="Path to the properties file")
    args = parser.parse_args()

    interface = InterfaceReqHandler(args.port, args.properties_file)
    interface.run()
