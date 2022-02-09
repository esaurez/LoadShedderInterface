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

def construct_hsv_histogram():
    hsvHistogram = mapping_capnp.HsvHistogram.new_message()
    num_bins = 8
    hsvHistogram.binSize = 32
    hsvHistogram.totalCountedPixels = 1000
    num_colors = 1
    value_hists = hsvHistogram.init("colorHistograms", num_colors)
    color_ranges = hsvHistogram.init("colorRanges", num_colors)

    for color_idx in range(num_colors):
        color_range = color_ranges[color_idx]
        value_hist = value_hists[color_idx]
        sat_hists = value_hist.init("valueBins", num_bins)
        for sat_hist in sat_hists:
            counts = sat_hist.init("counts", num_bins)
            for b in range(num_bins):
                counts[b] = 2

        hue_ranges = color_range.init("ranges", 2)
        hue_ranges[0].hueBegin = 0
        hue_ranges[0].hueBegin = 10
        hue_ranges[1].hueBegin = 170
        hue_ranges[1].hueBegin = 180

    return hsvHistogram

def build_util_req(vid_idx):
    util_msg = messages_capnp.UtilityMessage.new_message()
    util_msg.messageType = messages_capnp.UtilityMessage.Type.utilityRequest
    util_req = messages_capnp.UtilityRequest.new_message()
    util_req.mode= "all_cdf"
    util_req.videoIdx = vid_idx

    feats = mapping_capnp.Features.new_message()
    features = feats.init("feats", 1)
    feature = features[0]
    feature.type = mapping_capnp.Feature.Type.hsvHistogram

    feature.feat.hsvHisto = construct_hsv_histogram()
   
    util_req.feats = feats

    util_msg.utilityRequest = util_req
    return util_msg

def main(model):
    print (interface.handle_msg(build_util_threshold_req([0,7], [0,0.8])))
    util_req = build_util_req(0)
    start_ts = time.time()
    print (interface.handle_msg(util_req))
    end_ts = time.time()
    print ("Time taken to execute get_util = ", (end_ts - start_ts)*1000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pf", "--properties_file", dest="properties_file", required=True, type=str, help="Path to the properties file")
    args = parser.parse_args()
    interface = InterfaceReqHandler(0, args.properties_file)

    main(interface)
