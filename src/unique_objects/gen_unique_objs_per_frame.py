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
        obj = int(s[0])
        start = int(s[1])
        end = int(s[2])
        for idx in range(start, end+1):
            if idx not in frames:
                frames[idx] = []
            frames[idx].append(obj)

    with open(join(outdir, "unique_objs_per_frame.txt"), "w") as f:
        f.write("#Ignore first line. Each line has frame idx with variable number of Object IDS.\n")
        for frame in sorted(frames.keys()):
            f.write("%d\t"%frame)
            for obj in frames[frame]:
                f.write("%d\t"%obj)
            f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzing the distribution of Pixel Fraction")
    parser.add_argument("-I", "--obj-file", dest="obj_file", help="Path to file containing object lifetimes")
    parser.add_argument("-O", "--output-dir", dest="outdir", help="Output directory")

    args = parser.parse_args()
    main(args.obj_file, args.outdir)
