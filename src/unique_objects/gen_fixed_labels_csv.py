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

def main(obj_file, outdir):
    with open(obj_file) as f:
        lines = f.readlines()[1:]
    frames = {}
    for l in lines:
        s = l.split()
        start = int(s[1])
        end = int(s[2])
        for idx in range(start, end+1):
            frames[idx] = True

    with open(join(outdir, "updated_frames_labels.csv"), "w") as f:
        f.write("frame_idx,label\n")
        for frame in sorted(frames.keys()):
            f.write("%d,1\n"%frame)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzing the distribution of Pixel Fraction")
    parser.add_argument("-I", "--obj-file", dest="obj_file", help="Path to file containing object lifetimes")
    parser.add_argument("-O", "--output-dir", dest="outdir", help="Output directory")

    args = parser.parse_args()
    main(args.obj_file, args.outdir)
