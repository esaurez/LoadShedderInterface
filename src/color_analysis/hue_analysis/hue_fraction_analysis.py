import argparse
import os, sys
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../../"))
import model.build_model
import python_server.mapping_features
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from os import listdir
from os.path import basename, join, isdir, isfile
import math
import numpy as np
import yaml
from capnp_serial import mapping_capnp

def calculate_hue_fraction(observation, color):
    feature = observation.feats.feats[0]
    if feature.type == mapping_capnp.Feature.Type.fullColorHistogram:
        whole_histo = feature.feat.wholeHisto.histo
        hues = whole_histo.counts[0].count
        total_pixels = whole_histo.totalCountedPixels

        hue_count = 0
        for (low, high) in color:
            for idx in range(low, high):
                hue_count += hues[idx]
        hf = hue_count/float(total_pixels) 
        return hf
    elif feature.type == mapping_capnp.Feature.Type.hsvHistogram:
        hists = feature.feat.hsvHisto.colorHistograms 
        fg_pixels = feature.feat.hsvHisto.totalCountedPixels
        if len(hists) > 1:
            print ("Multi color not supported rn. Exiting.")
            exit(1)
        hist = hists[0]
        
        total_color_pixels = 0
        for row_idx in range(len(hist.valueBins)):
            row = hist.valueBins[row_idx]
            for col_idx in range(len(row.counts)):
                col = row.counts[col_idx]
                total_color_pixels += col
        hue_fraction = total_color_pixels/fg_pixels
        return hue_fraction

def main(training_conf_dir):
    with open(join(training_conf_dir, "conf.yaml")) as f:
        conf = yaml.safe_load(f)
    colors = {}
    for hue_bin in conf["hue_bins"]:
        name = hue_bin["name"]
        colors[name] = []
        for r in hue_bin["ranges"]:
            colors[name].append((r["start"], r["end"]))

    training_dir = conf["training_dir"]
    vids = [d for d in listdir(training_dir) if isdir(join(training_dir, d))]
    for vid in vids:
        vid_dir = join(training_dir, vid)
        bin_file = [join(vid_dir, f) for f in listdir(vid_dir) if isfile(join(vid_dir, f)) and f.endswith(".bin")][0]
        # Extracting the features from training data samples
        observations = python_server.mapping_features.read_samples(bin_file)
        frame_id=0

        color_hfs = {}
        for color in colors:
            color_hfs[color] = []

        for observation in observations:
            if frame_id < 1:
                frame_id += 1
                continue

            label = observation.label
            if not label:
                frame_id += 1
                continue
            
            feature = observation.feats.feats[0]
            for color in colors:
                hf = calculate_hue_fraction(observation, colors[color])
                color_hfs[color].append(hf)
            frame_id += 1
        print (vid, end="\t")
        for color in colors:
            print (min(color_hfs[color]), end="\t")
        print ("")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzing what should be the cutoff Hue Fraction.")
    parser.add_argument("-C", dest="training_conf_dir", help="Path to training conf folder")

    args = parser.parse_args()
    main(args.training_conf_dir)

