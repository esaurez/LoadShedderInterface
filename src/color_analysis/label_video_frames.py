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

def main(bin_file, video_file, outdir, start_frame, util_csv):
    util_df = pd.read_csv(util_csv)
    vid_name = basename(bin_file)[:-4]
    util_df = util_df[util_df["video_name"] == vid_name]
    frame_to_util = {}
    for idx, row in util_df.iterrows():
        frame_to_util[row["frame_id"]] = row["util"]

    # Extracting the features from training data samples
    observations = python_server.mapping_features.read_samples(bin_file)
    #for idx in range(len(observations)):
    #    print (idx)

    frame_size = None
    fps = 1

    # Reading the video frame by frame
    vidcap = cv2.VideoCapture(video_file)
    vidwriter = None
    frame_idx = 0
    ret, frame = vidcap.read()
    while ret:
        print ("Processing frame_idx ", frame_idx)
        if frame_size == None:
            frame_size = (int(vidcap.get(3)), int(vidcap.get(4)))
            outfile_name = basename(video_file)[:-4]+"_labeled.avi"
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            vidwriter = cv2.VideoWriter(join(outdir, outfile_name), fourcc, fps, frame_size)

        if frame_idx >= start_frame:
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)  
            # fontScale
            fontScale = 1
            # Blue color in BGR
            color = (255, 0, 0)
            # Line thickness of 2 px
            thickness = 2

            cv2.putText(frame, "Utility = %.2f"%frame_to_util[frame_idx], org, font, fontScale, color, thickness, cv2.LINE_AA)
            vidwriter.write(frame)

        ret, frame = vidcap.read()
        frame_idx += 1

    vidwriter.release()
    vidcap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzing the distribution of Pixel Fraction")
    parser.add_argument("-B", "--bin-file", dest="bin_file", help="Bin file")
    parser.add_argument("-V", "--video-file", dest="video_file", help="Video file")
    parser.add_argument("-O", "--output-dir", dest="outdir", help="Output directory")
    parser.add_argument("-U", "--util-csv", dest="util_csv", help="CSV file containing the utility values")
    parser.add_argument("--start-frame", dest="start_frame", help="Index of the frame to start labeling", type=int)

    args = parser.parse_args()
    main(args.bin_file, args.video_file, args.outdir, args.start_frame, args.util_csv)
