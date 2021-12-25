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


class InterfaceReqHandler:

    def __init__(self, port, properties_file):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:%s" % port)

        config = ConfigObj(properties_file)
       #featureCorrespondingBinSize = list(map(int, config["featureCorrespondingBinSize"]))
        splitvalues = config["splitvalues"] 
        generatedModelPath = config["generatedModelPath"]
        #model.build_model.init_shedding(generatedModelPath, featureCorrespondingBinSize) 
        model.build_model.init_shedding(generatedModelPath) # hsb

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
    def compute_util_threshold(self, drop_ratio, mode):
        return model.build_model.get_utility_threshold(drop_ratio, mode) #! Returns two values now. First one is the threshold. 

    ## Stuttgart folks implement this function ##
    def compute_utility(self, features, mode):
        featureList = mapping_features.map_features(features) # does only need the feature, not the HistogramBinSize any longer.
        return model.build_model.get_utility(featureList, mode) # needs the mode now as second feature. Mode is written in the config file

    def handle_util_threshold_req(self, util_threshold_req):
        drop_ratio = util_threshold_req.dropRatio
        mode = util_threshold_req.mode

        # Function call to compute the util threshold for given drop ratio
        util_threshold, th_ratio = self.compute_util_threshold(drop_ratio, mode)

        reply = messages_capnp.UtilityMessage.new_message()
        reply.messageType = "utilityThresholdResponse"
        reply.init("utilityThresholdResponse")
        reply.utilityThresholdResponse.threshold = util_threshold
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
    with open('/home/surveillance/test_feature_request.bin', 'rb') as f:
        util_msg = messages_capnp.UtilityMessage.from_bytes(f.read())
        print (util_msg)
