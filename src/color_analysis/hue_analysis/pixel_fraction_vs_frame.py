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
import hue_fraction_analysis

def main(training_conf_dir, outdir):
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
        print (vid)
        vid_dir = join(training_dir, vid)
        bin_file = [join(vid_dir, f) for f in listdir(vid_dir) if isfile(join(vid_dir, f)) and f.endswith(".bin")][0]
        # Extracting the features from training data samples
        observations = python_server.mapping_features.read_samples(bin_file)
        frame_id=0

        color_hfs = {}
        for color in colors:
            color_hfs[color] = []

        data = {}
        for c in colors:
            data[c] = {True:[[],[]], False:[[],[]]}

        for frame_id in range(len(observations)):
            observation = observations[frame_id]
            if frame_id == 0:
                continue

            label = observation.label

            for color in colors:
                hf = hue_fraction_analysis.calculate_hue_fraction(observation, colors[color])

                data[color][label][0].append(frame_id)
                data[color][label][1].append(hf)

        # Now plot
        for c in colors:
            plt.close()
            fig, ax = plt.subplots(figsize=(24,6))
            ax.bar(data[c][True][0], data[c][True][1], color="red", label="+ve")
            ax.bar(data[c][False][0], data[c][False][1], color="blue", label="-ve")
            ax.grid()
            ax.legend()
            ax.set_xlabel("Frame ID")
            ax.set_ylabel("Hue Fraction (%s)"%c)
            ax.set_xlim([0, len(data[c][True][0])+len(data[c][False][0])])

            min_tick = min(min(data[c][True][0]),min(data[c][False][0]))
            max_tick = max(max(data[c][True][0]),max(data[c][False][0]))

            ax.set_xticks(np.arange(min_tick, max_tick,100))
            ax.set_xticklabels(np.arange(min_tick, max_tick,100), rotation=90)
            
            fig.savefig(join(outdir, "hf_%s_%s.png"%(vid, c)), bbox_inches="tight")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzing what should be the cutoff Hue Fraction.")
    parser.add_argument("-C", dest="training_conf_dir", help="Path to training conf folder")
    parser.add_argument("-O", dest="outdir", help="Path to output dir")

    args = parser.parse_args()
    main(args.training_conf_dir, args.outdir)

