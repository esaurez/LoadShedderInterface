import argparse
import os, sys
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../"))
sys.path.insert(0, os.path.join(script_dir, "../../"))
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

def main(video_file, bin_file, outdir):
    frame_to_label = read_pixel_hsv_from_video.map_frame_to_label(bin_file)

    frame_count = 0
    cap = cv2.VideoCapture(video_file)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc=fourcc, fps=1, frameSize=)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_to_label[frame_count]:
            out.write(frame)

        frame_count += 1
        if frame_count % 10 == 0:
            print ("Processed %d frames"%frame_count)

    out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzing the distribution of Pixel Fraction")
    parser.add_argument("-V", "--video-file", dest="video_file", help="Path to video file")
    parser.add_argument("-B", "--bin-file", dest="bin_file", help="Path to the bin file")
    parser.add_argument("-O", dest="outdir", help="Directory to store output plots")
    
    args = parser.parse_args()
    main(args.video_file, args.bin_file, args.outdir)
