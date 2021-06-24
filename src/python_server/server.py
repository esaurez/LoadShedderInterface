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
    fullHistogramBinSize = 1

    def __init__(self, port, properties_file):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:%s" % port)

        config = ConfigObj(properties_file)
        self.fullHistogramBinSize = int(config["fullHistogramBinSize"])
        featureCorrespondingBinSize = list(map(int, config["featureCorrespondingBinSize"]))
        generatedModelPath = config["generatedModelPath"]
        model.build_model.init_shedding(generatedModelPath, featureCorrespondingBinSize)

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
    def compute_util_threshold(self, drop_ratio):
        # return 0.5 # TODO
        return model.build_model.get_utility_threshold(drop_ratio)

    ## Stuttgart folks implement this function ##
    def compute_utility(self, features):
        # return 0.6 # TODO
        featureList = mapping_features.map_features(features, self.fullHistogramBinSize)
        return model.build_model.get_utility(featureList)

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
        #features = util_req.feats.feats
        features = util_req.feats

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
    parser.add_argument("--pf", "--properties_file", dest="properties_file", required=True, type=str,
                        help="Path to the properties file")
    args = parser.parse_args()

    interface = InterfaceReqHandler(args.port, args.properties_file)
    interface.run()
