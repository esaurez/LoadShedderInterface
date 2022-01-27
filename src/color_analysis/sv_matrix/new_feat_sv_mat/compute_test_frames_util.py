import argparse
import yaml
import os, sys
from os import listdir
from os.path import join, isfile, basename
from concurrent.futures import ProcessPoolExecutor
import concurrent
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../../../"))
import python_server.mapping_features
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import extract_sv_matrix_from_bin

max_frames=None

def dump_sv_mat(mat, outfile):
    with open(outfile, "w") as f:
        for row in mat:
            for col in row:
                f.write("%f "%col)
            f.write("\n")

def read_utils(util_files):
    utils = []
    for util_file in util_files:
        U = [] # per file matrix
        with open(util_file) as f:
            for line in f.readlines():
                U.append([])
                s = line.split()
                for x in s:
                    U[-1].append(float(x))
        utils.append(U)
    return utils

def compute_util(frame_mat, util_mat):
    if frame_mat == None: # due to low PF
        return 0
    util = 0
    for row in range(len(frame_mat)):
        for col in range(len(frame_mat[row])):
            util += frame_mat[row][col]*util_mat[row][col]
    return util

def main(training_conf_file, num_bins, bin_file, outdir, util_files, vid_name, all_frames, pf_threshold):
    utils = read_utils(util_files)
    ground_truth_frames = python_server.mapping_features.read_samples(bin_file)

    with open(training_conf_file) as f:
        training_conf = yaml.safe_load(f)
   
    training_split = training_conf["training_split"]
    if len(util_files) != len(training_conf["hue_bins"]):
        print ("Mismatch between num util files provided (%d) and num colors in training conf (%d)"%(len(util_files), len(training_conf["hue_bins"])))
        exit(1)

    # Creating the ProcessPoolExecutor
    executor = ProcessPoolExecutor(max_workers=32)

    conf_colors = extract_sv_matrix_from_bin.get_conf_colors(training_conf)
    colors = training_conf["hue_bins"]
    sv_mat_list = [[] for idx in range(len(colors))]

    num_total_frames = len(ground_truth_frames)
    num_training_frames = int(training_split*num_total_frames)

    futures = []
    for frame_idx in range(num_total_frames):
        if (not all_frames) and frame_idx < num_training_frames:
            continue
        frame = ground_truth_frames[frame_idx]
        future = executor.submit(extract_sv_matrix_from_bin.get_sv_counts, frame, conf_colors, num_bins, pf_threshold)
        futures.append(future)

        if max_frames != None and frame_idx > max_frames:
            break

    frame_idx = 0
    raw_data = []

    for f in futures:
        label = ground_truth_frames[frame_idx].label
        count = ground_truth_frames[frame_idx].detections.totalDetections
        mats_list = f.result()

        for color_idx in range(len(mats_list)):
            util = compute_util(mats_list[color_idx], utils[color_idx])
            raw_data.append([colors[color_idx]["name"], count, label, frame_idx, util, vid_name])
        frame_idx += 1
    df = pd.DataFrame(raw_data, columns=["color", "count", "label", "frame_id", "utility", "vid_name"])
    df.to_csv(join(outdir, "frame_utils.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzing the distribution of Pixel Fraction")
    parser.add_argument("-B", dest="bin_file", help="Path to bin file")
    parser.add_argument("-A", dest="all_frames", help="Process all frames in video, not just testing frames", action="store_true", default=False)
    parser.add_argument("-V", dest="vid_name", help="Name of video")
    parser.add_argument("-C", dest="training_conf", help="Path to the training conf yaml")
    parser.add_argument("-O", dest="outdir", help="Path to output directory")
    parser.add_argument("--bins", dest="num_bins", help="Number of bins", type=int, default=16)
    parser.add_argument("--util-files", dest="util_files", nargs="*", help="Util files. Should be in the same order as colors in training conf")
    parser.add_argument("--pf-threshold", dest="pf_threshold", help="Min PF of given color (over pixels with sat>0 or val>0)", type=float, default=0.025)

    args = parser.parse_args()
    main(args.training_conf, args.num_bins, args.bin_file, args.outdir, args.util_files, args.vid_name, args.all_frames, args.pf_threshold)

