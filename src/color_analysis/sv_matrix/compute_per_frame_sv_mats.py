import argparse
import yaml
import os, sys
from os import listdir
from os.path import join, isfile, basename
from concurrent.futures import ProcessPoolExecutor
import concurrent
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../../"))
import python_server.mapping_features
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import extract_sv_matrix

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

def main(frame_dir, training_conf_file, outdir):
    with open(training_conf_file) as f:
        training_conf = yaml.safe_load(f)
   
    pf_threshold = training_conf["pf_threshold"]
    # Creating the ProcessPoolExecutor
    executor = ProcessPoolExecutor(max_workers=32)

    colors = training_conf["hue_bins"]
    if len(colors) > 1:
        print ("This script is only for single-color")
        exit(1)

    num_bins = training_conf["num_bins"]
    sv_mat_list = [[] for idx in range(len(colors))]

    # Assign a unique index to each
    frames = [join(frame_dir, f) for f in listdir(frame_dir) if isfile(join(frame_dir, f))]
    frames = sorted(frames, key = lambda x : int(basename(x)[:-4].split("_")[1]))

    futures = []
    for frame_idx in range(len(frames)):
        frame = frames[frame_idx]
        future = executor.submit(extract_sv_matrix.get_sv_counts, frame, colors, num_bins, pf_threshold)
        futures.append(future)

    frame_idx = 0
    raw_data = []

    for f in futures:
        mat = f.result()[0]

        with open(join(outdir, "frame_%d.txt"%(frame_idx)), "w") as f:
            for row in range(num_bins):
                for col in range(num_bins):
                    if mat == None:
                        f.write("0 ")
                    else:
                        f.write("%f "%(mat[row][col]))
                f.write("\n")

        frame_idx += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzing the distribution of Pixel Fraction")
    parser.add_argument("-F", dest="frame_dir", help="Path to directory containing hsv for each frame")
    parser.add_argument("-C", dest="training_conf", help="Path to the training conf yaml")
    parser.add_argument("-O", dest="outdir", help="Path to output directory")

    args = parser.parse_args()
    main(args.frame_dir, args.training_conf, args.outdir)

