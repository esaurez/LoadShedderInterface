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
import img2pdf

def main(video_file, bin_file, outdir):
    observations = python_server.mapping_features.read_samples(bin_file)

    jpgs = []
    positive_frame_idxs = []
    frame_idx = 0
    vidcap = cv2.VideoCapture(video_file)
    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if ret == False:
            break
        label = observations[frame_idx].label
        obj_count  = observations[frame_idx].detections.totalDetections
        if label:
            # font
            font = cv2.FONT_HERSHEY_SIMPLEX
            # org
            org = (50, 50)
            # fontScale
            fontScale = 1
            # Blue color in BGR
            color = (255, 0, 0)
            # Line thickness of 2 px
            thickness = 2
            # Using cv2.putText() method
            frame = cv2.putText(frame, "FrameIdx %d : %d objects"%(frame_idx, obj_count), org, font, fontScale, color, thickness, cv2.LINE_AA)

            outfile = join(outdir, "frame_%d.jpg"%frame_idx)
            cv2.imwrite(outfile, frame)
            jpgs.append(outfile)
            positive_frame_idxs.append(frame_idx)
        frame_idx += 1
    vidcap.release()

    with open(join(outdir, "combined.pdf"), "wb") as f:
        f.write(img2pdf.convert(jpgs))

    with open(join(outdir, "unique_objs_per_frame.txt"), "w") as f:
        f.write("#Ignore first line. Each line has frame idx with variable number of Object IDS.\n")
        for idx in positive_frame_idxs:
            f.write("%d\t\n"%idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzing the distribution of Pixel Fraction")
    parser.add_argument("-V", "--video-file", dest="video_file", help="Video file")
    parser.add_argument("-B", "--bin-file", dest="bin_file", help="BIN file")
    parser.add_argument("-O", "--output-dir", dest="outdir", help="Output directory")

    args = parser.parse_args()
    main(args.video_file, args.bin_file, args.outdir)
