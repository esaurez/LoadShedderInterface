import argparse
import yaml
import os, sys
from os import listdir
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
from os.path import basename, join, isdir, isfile
import math
import cv2
import img2pdf

def write(video_file, frames_to_write, vid_name, outdir):
    frame_to_util = {}
    for page_no in range(len(frames_to_write)):
        (frame_idx, util) = frames_to_write[page_no]
        frame_to_util[frame_idx] = util

    frame_to_jpg = {}

    frame_idx = 0
    vidcap = cv2.VideoCapture(video_file)
    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if ret == False:
            break
        if frame_idx in frame_to_util:
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
            frame = cv2.putText(frame, "FrameIdx %d : Util = %.3f" %(frame_idx, frame_to_util[frame_idx]), org, font, fontScale, color, thickness, cv2.LINE_AA)

            outfile = join(outdir, "frame_%d.jpg"%frame_idx)
            cv2.imwrite(outfile, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
            frame_to_jpg[frame_idx] = outfile
        frame_idx += 1
    vidcap.release()

    jpgs = [frame_to_jpg[fidx] for (fidx, u) in frames_to_write]

    with open(join(outdir, "pos_utils_%s.pdf"%vid_name), "wb") as f:
        f.write(img2pdf.convert(jpgs))

    for j in jpgs:
        os.remove(j)

def main(training_dir, util_file, outdir):
    training_conf_file = join(training_dir, "conf.yaml")
    with open(training_conf_file) as f:
        training_conf = yaml.safe_load(f)
    tdir = training_conf["training_dir"]

    df = pd.read_csv(util_file)
    g = df.groupby("vid_name")
    for vid in g.groups.keys():
        gdf = g.get_group(vid)
        frames_to_write = []
        for idx, row in gdf.iterrows():
            label = row["label"]
            util = row["utility"]
            frame_idx = row["frame_idx"]
            if label:
                frames_to_write.append((frame_idx, util))
            
        frames_to_write = sorted(frames_to_write, key = lambda x : x[1])

        vid_dir = join(tdir, vid)
        avi = [join(vid_dir, f) for f in listdir(vid_dir) if isfile(join(vid_dir, f)) and f.endswith(".avi")][0]

        write(avi, frames_to_write,  vid, outdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-C", dest="training_conf", help="Traiing conf dir")
    parser.add_argument("-U",dest="util_file", help="Util file (frame_utils.csv)")
    parser.add_argument("-O", "--output-dir", dest="outdir", help="Output directory")

    args = parser.parse_args()
    main(args.training_conf, args.util_file, args.outdir)
