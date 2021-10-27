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
from os.path import basename, join, isfile, isdir
import math
import cv2
import numpy as np
import read_pixel_hsv_from_video
import plot_sv_weight_heatmap

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

def validate_inputs(frame_dirs, bin_files):
    if len(frame_dirs) != len(bin_files):
        print ("Number of frame_dirs (%d) does not match number of bin_files (%d)"%(len(frame_dirs), len(bin_files)))
        return False
    for frame_dir in frame_dirs:
        if not isdir(frame_dir):
            print ("Frame dir %s does not exist"%frame_dir)
            return False
    for bin_file in bin_files:
        if not isfile(bin_file):
            print ("Bin file %s does not exist"%bin_file)
            return False
    return True

def plot_sv_heatmap(util_matrix, total_matrix, max_weight, filter_color, filter_pixel_fraction, outdir, final_bin_size):
    max_val = 0
    for l in util_matrix:
        for row in range(len(util_matrix[l])):
            for col in range(len(util_matrix[l][row])):
                m = util_matrix[l][row][col]
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

    filter_info = "_"
    if filter_color:
        filter_info += "%s_%.2f"%(filter_color, filter_pixel_fraction)
    fig.savefig(join(outdir, "sv_weight_heatmap_PF%s_BIN_%d_CROSSVIDEO.png"%(filter_info, final_bin_size)), bbox_inches="tight")

def initialize_util_matrix(final_bin_size):
    util_matrix = {True:[], False:[]}
    for d in [True, False]:
        for row in range(final_bin_size):
            util_matrix[d].append([[] for i in range(final_bin_size)])
    return util_matrix

def main(frame_dirs, bin_files, outdir, final_bin_size, filter_color, filter_pixel_fraction, training_ratio, max_weight=None):
    if not validate_inputs(frame_dirs, bin_files):
        print ("Invalid input. Exiting")
        exit(1)

    # To store information across videos
    util_matrix = None
    total_matrix = None

    for idx in range(len(frame_dirs)):
        bin_file = bin_files[idx]
        frame_dir = frame_dirs[idx]

        video_name = basename(bin_file)[:-4]

        # Extracting the features from training data samples
        observations = python_server.mapping_features.read_training_file(bin_file, absolute_pixel_count=False)
        training_samples = python_server.mapping_features.read_samples(bin_file)

        num_frame_files = len([o for o in listdir(frame_dir) if isfile(join(frame_dir, o)) and o.startswith("frame_") and o.endswith(".txt")])

        if num_frame_files != len(training_samples):
            print ("Mismatch b/w num_frames(%d) and num_training_samples(%d) for video %s. Exiting."%(num_frame_files, len(training_samples), video_name))
            exit(1)

        num_training_frames = training_ratio * num_frame_files

        frame_id = 0
        for sample in training_samples:
            if frame_id > num_training_frames:
                break

            label = sample.label
            # Determine if Pixel Fraction is higher than threshold
            high_pf = plot_sv_weight_heatmap.is_pf_high(filter_color, filter_pixel_fraction, observations[frame_id])
            # Read the raw SV weights from frame directory
            sv_weights = plot_sv_weight_heatmap.read_frame_sv_weights(join(frame_dir, "frame_%d.txt"%frame_id))
            # Readjust the SV weights since we are not counting S=0 and V=0 pixels
            res = plot_sv_weight_heatmap.readjust_sv_weights(sv_weights)

            if res == False or high_pf == False:
                frame_id += 1
                continue
        
            # Initializing the utility matrix
            if util_matrix == None:
                util_matrix = initialize_util_matrix(final_bin_size)
                total_matrix = {True:0, False:0}

                # Check that the final_bin_size is a factor of original bin size
                orig_size = len(sv_weights)
                q = int (orig_size / final_bin_size)
                if final_bin_size * q != orig_size:
                    print ("Final bin size %d is not a factor of original bin size %d"%(final_bin_size, orig_size))

            # Aggregating the SV weights into the final SV bins
            aggr = plot_sv_weight_heatmap.aggregate_sv_weights(sv_weights, final_bin_size)

            # Now contributing to the average across frames
            for row in range(final_bin_size):
                for col in range(final_bin_size):
                    util_matrix[label][row][col].append(aggr[row][col])

            total_matrix[label] += 1

            frame_id += 1

    # Computing the average value for each bin
    for l in util_matrix:
        for row in range(len(util_matrix[l])):
            for col in range(len(util_matrix[l][row])):
                m = np.mean(util_matrix[l][row][col])
                util_matrix[l][row][col] = m

    # Plotting the SV heatmap
    plot_sv_heatmap(util_matrix, total_matrix, max_weight, filter_color, filter_pixel_fraction, outdir, final_bin_size)

    # The second part of the script builds a utility model from the (s,v) 2D array
    combined_util_matrix = []
    for row in range(len(util_matrix[True])):
        combined_util_matrix.append([])
        for col in range(len(util_matrix[True][row])):
            combined_util_matrix[row].append(util_matrix[True][row][col] - util_matrix[False][row][col])

    plot_sv_weight_heatmap.normalize_combined_util_matrix(combined_util_matrix)

    # Now need to normalize the util matrix over training data
    max_util = 0
    for idx in range(len(frame_dirs)):
        bin_file = bin_files[idx]
        frame_dir = frame_dirs[idx]
        video_name = basename(bin_file)[:-4]

        # Extracting the features from training data samples
        observations = python_server.mapping_features.read_training_file(bin_file, absolute_pixel_count=False)
        training_samples = python_server.mapping_features.read_samples(bin_file)
        num_frame_files = len([o for o in listdir(frame_dir) if isfile(join(frame_dir, o)) and o.startswith("frame_") and o.endswith(".txt")])
        num_training_frames = int(training_ratio * num_frame_files)

        # Iterating over the training part of each video
        for frame_id in range(num_training_frames):
            sample = training_samples[frame_id]
            label = sample.label

            # Determine if Pixel Fraction is higher than threshold
            high_pf = plot_sv_weight_heatmap.is_pf_high(filter_color, filter_pixel_fraction, observations[frame_id])
            sv_weights = plot_sv_weight_heatmap.read_frame_sv_weights(join(frame_dir, "frame_%d.txt"%frame_id))
            res = plot_sv_weight_heatmap.readjust_sv_weights(sv_weights)

            if res == False or high_pf == False:
                continue

            aggr = plot_sv_weight_heatmap.aggregate_sv_weights(sv_weights, final_bin_size)

            util = 0
            for row in range(len(combined_util_matrix)):
                for col in range(len(combined_util_matrix[row])):
                    util += combined_util_matrix[row][col] * aggr[row][col]
            max_util = max(util, max_util)

    util_amplification_factor = 1.0/max_util
    print (util_amplification_factor)
    for row in range(len(combined_util_matrix)):
        for col in range(len(combined_util_matrix[row])):
            combined_util_matrix[row][col] *= util_amplification_factor

    raw_data = []

    for idx in range(len(frame_dirs)):
        bin_file = bin_files[idx]
        frame_dir = frame_dirs[idx]
        video_name = basename(bin_file)[:-4]

        # Extracting the features from training data samples
        observations = python_server.mapping_features.read_training_file(bin_file, absolute_pixel_count=False)
        training_samples = python_server.mapping_features.read_samples(bin_file)

        num_frame_files = len([o for o in listdir(frame_dir) if isfile(join(frame_dir, o)) and o.startswith("frame_") and o.endswith(".txt")])

        num_training_frames = training_ratio * num_frame_files

        frame_id = 0
        for sample in training_samples:
            label = sample.label

            # Determine if Pixel Fraction is higher than threshold
            high_pf = plot_sv_weight_heatmap.is_pf_high(filter_color, filter_pixel_fraction, observations[frame_id])
            sv_weights = plot_sv_weight_heatmap.read_frame_sv_weights(join(frame_dir, "frame_%d.txt"%frame_id))
            res = plot_sv_weight_heatmap.readjust_sv_weights(sv_weights)

            if res == False or high_pf == False or frame_id < num_training_frames:
                frame_id += 1
                continue

            aggr = plot_sv_weight_heatmap.aggregate_sv_weights(sv_weights, final_bin_size)

            util = 0
            for row in range(len(combined_util_matrix)):
                for col in range(len(combined_util_matrix[row])):
                    util += combined_util_matrix[row][col] * aggr[row][col]

            if label:
                label = 1
            else:
                label = 0
            raw_data.append([frame_id, label, util, video_name])
            frame_id += 1

    df = pd.DataFrame(raw_data, columns=["frame_id", "label", "util", "video_name"])
    max_util = df["util"].max()
    df["util"] = df["util"]/float(max_util)
    plt.close()
    fig, ax = plt.subplots(figsize=(6,2))
    sns.boxplot(data=df, x="video_name", y="util", hue="label", ax=ax)
    ax.set_xlabel("Label of the frame")
    ax.set_ylabel("Utility")
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
    fig.savefig(join(outdir, "combined_util_BIN_%d_CROSSVIDEO.png"%(final_bin_size)), bbox_inches="tight")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzing the distribution of Pixel Fraction")
    parser.add_argument("-I", "--frame-dirs", dest="frame_dirs", help="Path to frame directory", nargs="+")
    parser.add_argument("-B", "--bin-files", dest="bin_files", help="Path to the bin file", nargs="+")
    parser.add_argument("-S", "--final-bin-size", dest="final_bin_size", type=int, help="Final bin size (has to be a factor of the bin size of raw frame data)")
    parser.add_argument("-O", dest="outdir", help="Directory to store output plots")
    parser.add_argument("-C", dest="color", help="Color to use for filtering via Pixel Fraction")
    parser.add_argument("-F", dest="pixel_fraction_threshold", type=float, help="Min pixel fraction threshold")
    parser.add_argument("-T", dest="training_ratio", type=float, help="Fraction of total video playtime to use for training")
    
    args = parser.parse_args()
    main(args.frame_dirs, args.bin_files, args.outdir, args.final_bin_size, args.color, args.pixel_fraction_threshold, args.training_ratio)
