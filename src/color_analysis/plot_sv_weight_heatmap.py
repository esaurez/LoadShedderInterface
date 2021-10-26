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
from os import listdir
from os.path import basename, join, isfile
import math
import cv2
import numpy as np
import read_pixel_hsv_from_video

# TODO : Fix this definition of colors everywhere. Centralize it in a constants.py file.
# The main distribution of color values is in mapping_features file
COLORS = ['red', 'orange','yellow','spring_green','green','ocean_green','light_blue','blue','purple','magenta']

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

def get_total_pixel_fraction(sv):
    total = 0
    for row in range(len(sv)):
        for col in range(len(sv[row])):
            total += sv[row][col]
    return total

def aggregate_sv_weights(sv_weights, final_bin_size):
    # Create a util matrix for aggregating
    aggr= []
    for row in range(final_bin_size):
        aggr.append([])
        for col in range(final_bin_size):
            aggr[row].append(0)

    # Ratio of original bin size to final bin size
    aggr_ratio = int(len(sv_weights)/final_bin_size)

    for row in range(len(sv_weights)):
        for col in range(len(sv_weights[row])):
            if row < 1 or col < 1:
                continue
            else:
                aggr_row = int (row / aggr_ratio)
                aggr_col = int (col/aggr_ratio)
                aggr[aggr_row][aggr_col] += sv_weights[row][col]
    return aggr

def readjust_sv_weights(sv):
    total = 0
    for row in range(1, len(sv)):
        for col in range(1, len(sv[row])):
            total += sv[row][col]

    for row in range(len(sv)):
        for col in range(len(sv[row])):
            if row == 0 or col == 0:
                sv[row][col] = 0
            elif total == 0:
                sv[row][col] = 0
            else:
                sv[row][col] /= total

    return True

def main(frame_dir, bin_file, outdir, final_bin_size, filter_color, filter_pixel_fraction, max_weight):
    video_name = basename(bin_file)[:-4]
    util_matrix = None
    total_matrix = None

            # Extracting the features from training data samples
    observations = python_server.mapping_features.read_training_file(bin_file, absolute_pixel_count=False)
    training_samples = python_server.mapping_features.read_samples(bin_file)

    num_frame_files = len([o for o in listdir(frame_dir) if isfile(join(frame_dir, o)) and o.startswith("frame_") and o.endswith(".txt")])

    if num_frame_files != len(training_samples):
        print ("Mismatch b/w num_frames(%d) and num_training_samples(%d). Exiting."%(num_frame_files, len(training_samples)))

    frame_id = 0
    for sample in training_samples:
        label = sample.label
        high_pf = True
        if filter_color != None and filter_pixel_fraction != None:
            observation = observations[frame_id]
            feature_list = observation[0:-1]
            color_position = COLORS.index(filter_color)
            pixel_fraction = feature_list[color_position]
            if pixel_fraction < filter_pixel_fraction:
                high_pf = False

        sv_weights = read_frame_sv_weights(join(frame_dir, "frame_%d.txt"%frame_id))
        res = readjust_sv_weights(sv_weights)

        if res == False or high_pf == False:
            frame_id += 1
            continue
    
        # Initializing the utility matrix
        if util_matrix == None:
            util_matrix = {True:[], False:[]}
            for d in [True, False]:
                for row in range(final_bin_size):
                    util_matrix[d].append([[] for i in range(final_bin_size)])

            total_matrix = {True:0, False:0}

            # Check that the final_bin_size is a factor of original bin size
            orig_size = len(sv_weights)
            q = int (orig_size / final_bin_size)
            if final_bin_size * q != orig_size:
                print ("Final bin size %d is not a factor of original bin size %d"%(final_bin_size, orig_size))

        aggr = aggregate_sv_weights(sv_weights, final_bin_size)

        # Now contributing to the average across frames
        for row in range(final_bin_size):
            for col in range(final_bin_size):
                util_matrix[label][row][col].append(aggr[row][col])

        total_matrix[label] += 1

        frame_id += 1

    max_val = 0
    for l in util_matrix:
        for row in range(len(util_matrix[l])):
            for col in range(len(util_matrix[l][row])):
                #print (len(util_matrix[l][row][col]))
                m = np.mean(util_matrix[l][row][col])
                util_matrix[l][row][col] = m
                max_val = max(max_val, m)

    plt.close()
    rows = 1
    cols = 2
    fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    idx = 0
    for normalize in [True]:
        for label in util_matrix:
            row = int(idx / cols)
            col = idx - row*cols
            if rows > 1:
                ax = axs[row][col]
            else:
                ax = axs[col]

            if normalize:
                if max_weight:
                    vmax = max_weight
                else:
                    vmax = max_val
                normalized_str = "norm"
            else:
                normalized_str = ""
                vmax = None

            if label:
                label_str = "+ve frames"
            else:
                label_str = "-ve frames"

            sns.heatmap(util_matrix[label], ax=ax, cmap="BuPu", vmin=0, vmax=vmax)

            plot_label = "%s ; %s ; %d samples"%(label_str, normalized_str, total_matrix[label])
            ax.text(.5,.9, plot_label, horizontalalignment='center', transform=ax.transAxes)
            idx += 1

    fig.suptitle("Video file = %s ; Bins = %d"%(video_name, final_bin_size))
    filter_info = "_"
    if filter_color:
        filter_info += "%s_%.2f"%(filter_color, filter_pixel_fraction)
    fig.savefig(join(outdir, "sv_weight_heatmap_PF%s_BIN_%d_%s.png"%(filter_info, final_bin_size, video_name)), bbox_inches="tight")

    # The second part of the script builds a utility model from the (s,v) 2D array
    combined_util_matrix = []
    for row in range(len(util_matrix[True])):
        combined_util_matrix.append([])
        for col in range(len(util_matrix[True][row])):
            combined_util_matrix[row].append(util_matrix[True][row][col] - util_matrix[False][row][col])

    # Normalize
    min_val = 1.0
    for row in range(len(combined_util_matrix)):
        for col in range(len(combined_util_matrix[row])):
            min_val = min(min_val, combined_util_matrix[row][col])
    if min_val < 0:
        for row in range(len(combined_util_matrix)):
            for col in range(len(combined_util_matrix[row])):
                combined_util_matrix[row][col] += (0 - min_val)
    max_val = 0.0
    for row in range(len(combined_util_matrix)):
        for col in range(len(combined_util_matrix[row])):
            max_val = max(max_val, combined_util_matrix[row][col])

    for row in range(len(combined_util_matrix)):
        for col in range(len(combined_util_matrix[row])):
            combined_util_matrix[row][col] /= max_val

    raw_data = []
    frame_id = 0
    for sample in training_samples:
        label = sample.label

        high_pf = True
        if filter_color != None and filter_pixel_fraction != None:
            observation = observations[frame_id]
            feature_list = observation[0:-1]
            color_position = COLORS.index(filter_color)
            pixel_fraction = feature_list[color_position]
            if pixel_fraction < filter_pixel_fraction:
                high_pf = False

        sv_weights = read_frame_sv_weights(join(frame_dir, "frame_%d.txt"%frame_id))
        res = readjust_sv_weights(sv_weights)

        if res == False or high_pf == False:
            frame_id += 1
            continue

        aggr = aggregate_sv_weights(sv_weights, final_bin_size)

        util = 0
        for row in range(len(combined_util_matrix)):
            for col in range(len(combined_util_matrix[row])):
                util += combined_util_matrix[row][col] * aggr[row][col]

        if label:
            label = 1
        else:
            label = 0
        raw_data.append([frame_id, label, util])
        frame_id += 1

    df = pd.DataFrame(raw_data, columns=["frame_id", "label", "util"])
    max_util = df["util"].max()
    df["util"] = df["util"]/float(max_util)
    plt.close()
    fig, ax = plt.subplots(figsize=(6,2))
    sns.boxplot(data=df, x="label", y="util", ax=ax)
    ax.set_xlabel("Label of the frame")
    ax.set_ylabel("Utility")
    fig.savefig(join(outdir, "combined_util_BIN_%d_%s.png"%(final_bin_size, video_name)), bbox_inches="tight")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzing the distribution of Pixel Fraction")
    parser.add_argument("-I", "--frame-dir", dest="frame_dir", help="Path to frame directory")
    parser.add_argument("-B", "--bin-file", dest="bin_file", help="Path to the bin file")
    parser.add_argument("-S", "--final-bin-size", dest="final_bin_size", type=int, help="Final bin size (has to be a factor of the bin size of raw frame data)")
    parser.add_argument("-O", dest="outdir", help="Directory to store output plots")
    parser.add_argument("-C", dest="color", help="Color to use for filtering via Pixel Fraction")
    parser.add_argument("-F", dest="pixel_fraction_threshold", type=float, help="Min pixel fraction threshold")
    parser.add_argument("--max-weight", dest="max_weight", type=float, help="Max weight to use for heatmap normalization")
    
    args = parser.parse_args()
    main(args.frame_dir, args.bin_file, args.outdir, args.final_bin_size, args.color, args.pixel_fraction_threshold, args.max_weight)
