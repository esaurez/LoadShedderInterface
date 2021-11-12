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

def main(video_file, time_secs, outdir):
    vidcap = cv2.VideoCapture(video_file)

    total_video_len_secs = 900
    fps = 30
    target_frame_secs = time_secs
    frame_no = target_frame_secs/float(total_video_len_secs)

    vidcap.set(1, time_secs*fps)
    frames_read = 0
    ret, frame = vidcap.read()
    while ret:
        if frames_read > 8*fps:
            break

        print ("Read frame ", frames_read)

        ret, frame = vidcap.read()

        cv2.imwrite(join(outdir, "frame_%d.jpg"%frames_read), frame)

        frames_read += 1
    vidcap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzing the distribution of Pixel Fraction")
    parser.add_argument("-V", "--video-file", dest="video_file", help="Video file")
    parser.add_argument("-T", "--time-secs", dest="time_secs", help="Time in seconds", type=int)
    parser.add_argument("-O", "--output-dir", dest="outdir", help="Output directory")

    args = parser.parse_args()
    main(args.video_file, args.time_secs, args.outdir)
