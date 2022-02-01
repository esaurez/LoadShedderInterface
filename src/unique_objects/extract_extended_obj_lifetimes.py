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
    obj_lifetime_frames = 10
    observations = python_server.mapping_features.read_samples(bin_file)

    frames_to_write = {} # to contain a set of frames to write
    for idx in range(len(observations)):
        obs = observations[idx]
        if obs.label:
            low = max(0, idx-int(obj_lifetime_frames/2))
            high = min(len(observations)-1, idx+int(obj_lifetime_frames/2))
            for i in range(low, high+1):
                frames_to_write[i] = observations[i].label

    jpgs = []
    vidcap = cv2.VideoCapture(video_file)
    frame_idx = 0
    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if ret == False:
            break
        if frame_idx in frames_to_write:
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
            frame = cv2.putText(frame, "FrameIdx %d ; GT: %s"%(frame_idx, str(frames_to_write[frame_idx])), org, font, fontScale, color, thickness, cv2.LINE_AA)

            outfile = join(outdir, "frame_%d.jpg"%frame_idx)
            cv2.imwrite(outfile, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
            jpgs.append(outfile)
        frame_idx += 1
    vidcap.release()

    with open(join(outdir, "extended_lifetimes.pdf"), "wb") as f:
        f.write(img2pdf.convert(jpgs))

    for jpg in jpgs:
        os.remove(jpg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzing the distribution of Pixel Fraction")
    parser.add_argument("-V", "--video-file", dest="video_file", help="Video file")
    parser.add_argument("-B", "--bin-file", dest="bin_file", help="BIN file")
    parser.add_argument("-O", "--output-dir", dest="outdir", help="Output directory")

    args = parser.parse_args()
    main(args.video_file, args.bin_file, args.outdir)
