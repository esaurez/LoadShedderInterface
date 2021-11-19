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

def brute_force_count_fg(f):
    count = 0
    for r in range(len(f)):
        for c in range(len(f[r])):
            (h,s,v) = f[r][c]
            if h == 0 and s == 0 and v == 0:
                pass
            else:
                count += 1
    return count

def get_total_frame_weights(f):
    total = 0
    for row in f:
        for col in row:
            total += col
    return total

def main(video_file, low_frame_idx, high_frame_idx, bin_file, outdir):
    frame_to_label = read_pixel_hsv_from_video.map_frame_to_label(bin_file)

    # Defining the bins for Saturation and Value
    sat_bins = 16
    val_bins = 16
    sat_bin_size = 256 / sat_bins
    val_bin_size = 256 / val_bins

    # Creating the data structures for holding counts of pixels
    data = {True: [], False: []}
    total = {True: 0, False: 0}

    # Initializing the data structures holding pixel counts
    for d in [True, False]:
        for idx in range(val_bins):
            data[d].append([0 for i in range(sat_bins)])
    
    frame_count = 0
    for f in read_pixel_hsv_from_video.get_frames(video_file):

        frame_weightage = []
        for idx in range(val_bins):
            frame_weightage.append([0 for i in range(sat_bins)])
        total_pixels = 0

        if frame_count not in frame_to_label:
            frame_count += 1
            continue
        label = frame_to_label[frame_count]

        print ("Processing frame %d"%frame_count)
        row = 0

        # Counting foreground and background pixels
        fg_count = 0
        bg_count = 0 

        while row < len(f):
            col = 0

            last_col = 0 # Last value of column that was processed
            in_bg = True # Whether current traversal is through background pixels

            while col < len(f[row]):
                (h, s, v) = f[row][col]
                if s==0 and v==0 and h==0: # background pixel
                    in_bg = True
                    last_col = col
                else:
                    if in_bg: # If traversal was assuming background pixels
                        # Process all pixels from last_col + 1
                        for c in range(last_col+1, col+1):
                            (h, s, v) = f[row][c]
                            if not (h == 0 and s == 0 and v == 0):
                                fg_count += 1
                                if (h >= 0 and h <= 10) or (h >= 170 and h <= 180):
                                    val_idx = int(v / val_bin_size)
                                    sat_idx = int(s / sat_bin_size)
                                    frame_weightage[val_idx][sat_idx] += 1
                                    total_pixels += 1

                        in_bg = False
                        last_col = col

                    else:
                        (h, s, v) = f[row][col]
                        if not (h == 0 and s == 0 and v == 0):
                            fg_count += 1
                            if (h >= 0 and h <= 10) or (h >= 170 and h <= 180):
                                val_idx = int(v / val_bin_size)
                                sat_idx = int(s / sat_bin_size)
                                frame_weightage[val_idx][sat_idx] += 1
                                total_pixels += 1
                            
                        in_bg = False
                        last_col = col

                # Increment the value of col
                if in_bg:
                    col += 2 # Jump over pixels if in 
                else:
                    col += 1 # If in foreground, don't miss pixels

            row += 1

        with open(join(outdir, "frame_%d.txt"%frame_count), "w") as f:
            for row in range(val_bins):
                for col in range(sat_bins):
                    frame_weightage[row][col] /= float(total_pixels)
                    f.write(str(frame_weightage[row][col])+" ")
                f.write("\n")

        t = get_total_frame_weights(frame_weightage)
        if t < 0.999:
            print ("Total weight %f != 1. Exiting."%t)
            exit(0)

        frame_count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzing the distribution of Pixel Fraction")
    parser.add_argument("-V", "--video-file", dest="video_file", help="Path to video file")
    parser.add_argument("-B", "--bin-file", dest="bin_file", help="Path to the bin file")
    parser.add_argument("-L", "--low-frame-idx", dest="low_frame_idx", help="Low index for frame", type=int, default=0)
    parser.add_argument("-H", "--high-frame-idx", dest="high_frame_idx", help="High index for frame", type=int, default=900)
    parser.add_argument("-O", dest="outdir", help="Directory to store output plots")
    
    args = parser.parse_args()
    main(args.video_file, args.low_frame_idx, args.high_frame_idx, args.bin_file, args.outdir)
