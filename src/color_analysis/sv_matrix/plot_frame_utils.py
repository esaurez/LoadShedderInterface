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

def get_sv_counts(frame, colors, num_bins):
    bin_size = 256/float(num_bins)
    sv_mats = {}
    total_pixels = {}
    for color_idx in range(len(colors)):
        sv_mats[color_idx] = [[0 for col in range(num_bins)] for row in range(num_bins)]
        total_pixels[color_idx] = 0

    with open(frame) as f:
        for line in f.readlines():
            s = line.split()
            hue = int(s[0])
            sat = int(s[1])
            val = int(s[2])

            if sat == 0 or val == 0:
                continue

            for color_idx in range(len(colors)):
                color = colors[color_idx]
                pixel_matches = False
                for hue_range in color["ranges"]:
                    if hue >= hue_range["start"] and hue <= hue_range["end"]:
                        pixel_matches = True
                        break
                if pixel_matches:
                    sat_idx = int(sat/bin_size)
                    val_idx = int(val/bin_size)
                    sv_mats[color_idx][val_idx][sat_idx] += 1
                    total_pixels[color_idx] += 1

    # Now normalize
    for color_idx in range(len(colors)):
        for row in range(num_bins):
            for col in range(num_bins):
                if total_pixels[color_idx] == 0:
                    sv_mats[color_idx][row][col] = 0
                else:
                    sv_mats[color_idx][row][col] /= float(total_pixels[color_idx])

    return sv_mats

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

def main(util_files, outdir):
    dfs = []
    for util_file in util_files:
        df = pd.read_csv(util_file)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)      
    df["label"] = df["label"].astype(str)
    df["label_color"] = df[["label", "color"]].agg('-'.join, axis=1)

    plt.close()
    fig, ax = plt.subplots()
    sns.ecdfplot(data=df, x="utility", hue="label_color")
    ax.set_xlabel("Utility of frame")
    fig.savefig(join(outdir, "util_cdf_CROSSVIDEO.png"), bbox_inches="tight")

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
    fig.savefig(join(outdir, "util_count.png"), bbox_inches="tight")

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
    fig.savefig(join(outdir, "combined_util_CROSSVIDEO.png"), bbox_inches="tight")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzing the distribution of Pixel Fraction")
    parser.add_argument("-U", dest="util_files", help="Path to per-video utility files", nargs="*")
    parser.add_argument("-O", dest="outdir", help="Output directory")

    args = parser.parse_args()
    main(args.util_files, args.outdir)
