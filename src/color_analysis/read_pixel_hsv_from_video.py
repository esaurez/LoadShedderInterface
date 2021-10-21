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

def get_frames(video_file, low_frame_idx=None, high_frame_idx=None):
    frame_count = 0
    cap = cv2.VideoCapture(video_file)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if low_frame_idx != None and frame_count < low_frame_idx:
            frame_count += 1
            continue
        elif high_frame_idx != None and frame_count > high_frame_idx:
            break

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_count += 1

        if frame_count % 10 == 0:
            print ("Processed %d frames"%frame_count)

        yield hsv_frame

def map_frame_to_label(bin_file):
    result = {} # frame_id to label
    frame_id = 0
    # Extracting the features from training data samples
    training_samples = python_server.mapping_features.read_samples(bin_file)
    for sample in training_samples:

        label = sample.label
        result[frame_id] = label

        frame_id += 1
    return result

def main(video_file, low_frame_idx, high_frame_idx, bin_file, outdir):
    frame_to_label = map_frame_to_label(bin_file)

    sat_bins = 10
    val_bins = 10
    sat_bin_size = 256 / sat_bins
    val_bin_size = 256 / val_bins

    data = {True: [], False: []}
    total = {True: 0, False: 0}

    for d in [True, False]:
        for idx in range(val_bins):
            data[d].append([0 for i in range(sat_bins)])
    
    frame_count = 0
    for f in get_frames(video_file, low_frame_idx, high_frame_idx):
        if frame_count not in frame_to_label:
            frame_count += 1
            continue
        label = frame_to_label[frame_count]

        print ("Processing frame %d"%frame_count)
        row = 0
        while row < len(f):
            col = 0
            while col < len(f[row]):
                (h, s, v) = f[row][col]
                
                if (s>0 and v>0) and ((h >= 0 and h <= 10) or (h >= 170 and h <= 180)):
                    val_idx = int(v / val_bin_size)
                    sat_idx = int(s / sat_bin_size)
                    data[label][val_idx][sat_idx] += 1
                    total[label] += 1
                col += 4
            row += 4
        frame_count += 1
    
    max_val = 0
    for label in data:
        if total[label] == 0:
            continue
        for r in range(len(data[label])):
            for c in range(len(data[label][r])):
                data[label][r][c] /= float(total[label])
                max_val = max(max_val, data[label][r][c])

    for label in data:
        plt.close()
        fig, ax = plt.subplots(figsize=(4,4))
        sns.heatmap(data=np.array(data[label]), ax=ax, cmap="BuPu", vmin=0, vmax=max_val)
        ax.set_xlabel("Saturation")
        ax.set_ylabel("Value")
        fig.savefig(join(outdir, "red_sv_heatmap_label_%s.png"%str(label)), bbox_inches="tight")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzing the distribution of Pixel Fraction")
    parser.add_argument("-V", "--video-file", dest="video_file", help="Path to video file")
    parser.add_argument("-B", "--bin-file", dest="bin_file", help="Path to the bin file")
    parser.add_argument("-L", "--low-frame-idx", dest="low_frame_idx", help="Low index for frame", type=int, default=0)
    parser.add_argument("-H", "--high-frame-idx", dest="high_frame_idx", help="High index for frame", type=int, default=900)
    parser.add_argument("-O", dest="outdir", help="Directory to store output plots")
    
    args = parser.parse_args()
    main(args.video_file, args.low_frame_idx, args.high_frame_idx, args.bin_file, args.outdir)
