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
from server import InterfaceReqHandler

from model.sv_matrix_model import *

def build_util_threshold_req(vid_idxs, drop_ratios):
    util_msg = messages_capnp.UtilityMessage.new_message()
    util_msg.messageType = messages_capnp.UtilityMessage.Type.utilityThresholdRequest
    util_threshold_req = messages_capnp.UtilityThresholdRequest.new_message()
    util_threshold_req.mode= "all_cdf"
    vidIdxs = util_threshold_req.init("dropRatios", len(vid_idxs))
    for idx in range(len(vid_idxs)):
        vidIdxs[idx].videoIdx = vid_idxs[idx]
        vidIdxs[idx].dropRatio = drop_ratios[idx]
    util_msg.utilityThresholdRequest = util_threshold_req
    return util_msg

def main(model):
    print (interface.handle_msg(build_util_threshold_req([0,7], [0,0.8])))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pf", "--properties_file", dest="properties_file", required=True, type=str, help="Path to the properties file")
    args = parser.parse_args()
    interface = InterfaceReqHandler(0, args.properties_file)

    main(interface)
