import argparse
import numpy as np
from os import listdir
from os.path import join, isfile, isdir
import yaml
import pandas as pd
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../color_analysis"))
sys.path.insert(0, os.path.join(script_dir, "../color_analysis/sv_matrix"))
sys.path.insert(0, os.path.join(script_dir, "../"))
import model.build_model
import python_server.mapping_features

def main(training_conf):
    conf_file = join(training_conf, "conf.yaml")
    with open(conf_file) as f:
        conf = yaml.safe_load(f)
    training_dir = conf["training_dir"]

    vids = [v for v in listdir(training_dir) if isdir(join(training_dir, v))]

    DATA = [[],[]]

    vid_data = {}

    for vid in vids:
        vid_dir = join(training_dir, vid)
        vid_data[vid] = [[],[]]
        bin_file = [join(vid_dir, f) for f in listdir(vid_dir) if isfile(join(vid_dir, f)) and f.endswith(".bin")][0]
        observations = python_server.mapping_features.read_samples(bin_file)
        if len(DATA[1]) == 0:
            DATA[1] = [0 for idx in range(len(observations))]
            DATA[0] = range(len(observations))
        for idx in range(len(observations)):
            o = observations[idx]
            if o.label:
                DATA[1][idx] += 1
                vid_data[vid][1].append(1)
            else:
                vid_data[vid][1].append(0)
            vid_data[vid][0].append(idx)

    plt.close()
    fontsize=20
    fig, ax = plt.subplots(figsize=(24,6))
    ax.plot(DATA[0], DATA[1])
    fontP = FontProperties()
    fontP.set_size(fontsize)
    ax.set_xlabel("Frame idx", fontsize=fontsize)
    ax.set_ylabel("Activity", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    fig.savefig("activity.png", bbox_inches="tight")

    plt.close()
    fig,ax = plt.subplots(figsize=(24,4))
    bottom = []
    for vid in vid_data:
        if len(bottom) == 0:
            bottom = np.array([0 for i in vid_data[vid][0]])
        vid_data[vid][1] = np.array(vid_data[vid][1])
        ax.bar(vid_data[vid][0], vid_data[vid][1], bottom=bottom)
        bottom = bottom + vid_data[vid][1]
    fig.savefig("stacked.png", bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", dest="training_conf", help="Path to training conf dir", required=True)
    args = parser.parse_args()
    main(args.training_conf)
