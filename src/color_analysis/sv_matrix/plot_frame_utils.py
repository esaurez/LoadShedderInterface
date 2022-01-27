import argparse
import yaml
import os, sys
from os import listdir
from os.path import join, isfile, basename
from concurrent.futures import ProcessPoolExecutor
import concurrent
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../../"))
import python_server.mapping_features
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

max_frames=None

def dump_sv_mat(mat, outfile):
    with open(outfile, "w") as f:
        for row in mat:
            for col in row:
                f.write("%f "%col)
            f.write("\n")

def read_utils(util_files):
    utils = []
    for util_file in util_files:
        U = [] # per file matrix
        with open(util_file) as f:
            for line in f.readlines():
                U.append([])
                s = line.split()
                for x in s:
                    U[-1].append(float(x))
        utils.append(U)
    return utils

def compute_util(frame_mat, util_mat):
    util = 0
    for row in range(len(frame_mat)):
        for col in range(len(frame_mat[row])):
            util += frame_mat[row][col]*util_mat[row][col]
    return util

def main(util_files, outdir, training_split):
    dfs = []
    test_dfs = []
    for util_file in util_files:
        df = pd.read_csv(util_file)
        num_training_frames = int(training_split*len(df))
        test_df = df[df["frame_id"] > num_training_frames]
        dfs.append(df)
        test_dfs.append(test_df)

    for test_only in [True, False]:
        if test_only:
            plot_name = "TESTFRAMES"
            df = pd.concat(test_dfs, ignore_index=True)      
        else:
            plot_name = "ALLFRAMES"
            df = pd.concat(dfs, ignore_index=True)      

        df["label"] = df["label"].astype(str)
        df["label_color"] = df[["label", "color"]].agg('-'.join, axis=1)

        plt.close()
        fig, ax = plt.subplots()
        sns.ecdfplot(data=df, x="utility", hue="label_color")
        ax.set_xlabel("Utility of frame")
        fig.savefig(join(outdir, "util_cdf_CROSSVIDEO_%s.png"%plot_name), bbox_inches="tight")

        plt.close()
        vid_grouping = df.groupby("vid_name")
        num_vids = len(df["vid_name"].unique())
        cols = 3
        rows = math.ceil(num_vids/float(cols))
        fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(24,8))
        vid_idx = 0
        for vid in vid_grouping.groups.keys():
            vid_df = vid_grouping.get_group(vid)
            row = int(vid_idx/cols)
            col = vid_idx - row*cols
            if rows == 1:
                ax = axs[col]
            else:
                ax = axs[row][col]

            sns.lineplot(data=vid_df, x="frame_id", y="utility", ax=ax, color="blue", style="color")
            ax2 = ax.twinx()
            sns.lineplot(data=vid_df, x="frame_id", y="count", ax=ax2, color="black")
            lim = 0.025
            #ax.set_ylim([0, lim])
            #ax.set_ylim([-1*lim/2, lim])
            ax.set_title(vid, fontsize=10)
            ax.set_ylabel("Frame utility")
            ax.yaxis.label.set_color("blue")
            ax2.set_ylabel("#target objects")
            ax2.yaxis.label.set_color("black")

            vid_idx += 1
        fig.savefig(join(outdir, "util_count_%s.png"%plot_name), bbox_inches="tight")

        color_grouping = df.groupby("color")
        num_colors = len(color_grouping.groups.keys())
        plt.close()
        fig, axs = plt.subplots(nrows=1, ncols=num_colors, figsize=(24,8))
        color_idx = 0
        for color in color_grouping.groups.keys():
            if num_colors==1:
                ax = axs
            else:
                ax = axs[color_idx]
            gdf = color_grouping.get_group(color)
            sns.boxplot(data=gdf, x="vid_name", y="utility", hue="label", ax=ax)
            ax.set_xlabel("Label of the frame")
            ax.set_ylabel("Utility for %s"%color)
            ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
            color_idx += 1
        fig.savefig(join(outdir, "combined_util_CROSSVIDEO_%s.png"%plot_name), bbox_inches="tight")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzing the distribution of Pixel Fraction")
    parser.add_argument("-U", dest="util_files", help="Path to per-video utility files", nargs="*")
    parser.add_argument("-O", dest="outdir", help="Output directory")
    parser.add_argument("-S", dest="training_split", type=float, help="Training split in fraction", required=True)

    args = parser.parse_args()
    main(args.util_files, args.outdir, args.training_split)
