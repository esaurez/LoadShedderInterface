import argparse
import os, sys
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../"))
import model.build_model
import python_server.mapping_features
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from os.path import basename, join
import math
import cv2
import numpy as np
import read_pixel_hsv_from_video

def read_frame_sv_weights(frame_file):
    sv = []
    with open(frame_file) as f:
        for  line in f.readlines():
            s = line.split()
            row = []
            for v in s:
                row.append(float(v))
            sv.append(row)
    return sv

def main(frame_dir, bin_file, outdir):
    video_name = basename(bin_file)[:-4]
    util_matrix = None

    training_samples = python_server.mapping_features.read_samples(bin_file)
    frame_id = 0
    for sample in training_samples:
        label = sample.label
        sv_weights = read_frame_sv_weights(join(frame_dir, "frame_%d.txt"%frame_id))

        if util_matrix == None:
            util_matrix = {True:[], False:[]}
            for d in [True, False]:
                for row in range(len(sv_weights)):
                    util_matrix[d].append([[] for i in range(len(sv_weights[row]))])

        for row in range(len(sv_weights)):
            for col in range(len(sv_weights[row])):
                if row < 1 or col < 1:
                    continue
                else:
                    util_matrix[label][row][col].append(sv_weights[row][col])

        frame_id += 1

    max_val = 0
    for l in util_matrix:
        for row in range(len(util_matrix[l])):
            for col in range(len(util_matrix[l][row])):
                m = np.mean(util_matrix[l][row][col])
                util_matrix[l][row][col] = m
                max_val = max(max_val, m)

    for normalize in [True, False]:
        for label in util_matrix:
            plt.close()
            fig,ax = plt.subplots(figsize=(4,4))
            if normalize:
                vmax = max_val
                normalized_str = "norm"
            else:
                normalized_str = ""
                vmax = None
            sns.heatmap(util_matrix[label], ax=ax, cmap="BuPu", vmin=0, vmax=vmax)
            fig.savefig(join(outdir, "sv_weight_heatmap_%s_label_%s_%s.png"%(normalized_str, str(label), video_name)), bbox_inches="tight")

    # The second part of the script builds a utility model from the (s,v) 2D array
    combined_util_matrix = []
    for row in range(len(util_matrix[True])):
        combined_util_matrix.append([])
        for col in range(len(util_matrix[True][row])):
            if row < 1 or col < 1:
                continue
            combined_util_matrix[row].append(util_matrix[True][row][col] - util_matrix[False][row][col])

    raw_data = []
    frame_id = 0
    for sample in training_samples:
        label = sample.label
        sv_weights = read_frame_sv_weights(join(frame_dir, "frame_%d.txt"%frame_id))
        util = 0
        for row in range(len(combined_util_matrix)):
            for col in range(len(combined_util_matrix[row])):
                if row < 1 or col < 1 :
                    continue
                util += combined_util_matrix[row][col] * sv_weights[row][col]

        if label:
            label = 1
        else:
            label = 0
        raw_data.append([frame_id, label, util])
        frame_id += 1

    df = pd.DataFrame(raw_data, columns=["frame_id", "label", "util"])
    #print (df)
    plt.close()
    fig, ax = plt.subplots(figsize=(6,2))
    sns.scatterplot(data=df, x="frame_id", y="util", hue="label", ax=ax)
    ax.set_xlabel("Frame ID (1 frame per sec)")
    ax.set_ylabel("Utility")
    fig.savefig(join(outdir, "combined_util_%s.png"%video_name), bbox_inches="tight")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzing the distribution of Pixel Fraction")
    parser.add_argument("-I", "--frame-dir", dest="frame_dir", help="Path to frame directory")
    parser.add_argument("-B", "--bin-file", dest="bin_file", help="Path to the bin file")
    parser.add_argument("-O", dest="outdir", help="Directory to store output plots")
    
    args = parser.parse_args()
    main(args.frame_dir, args.bin_file, args.outdir)
