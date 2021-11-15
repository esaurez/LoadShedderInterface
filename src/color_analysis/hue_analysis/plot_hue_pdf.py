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
from os import listdir
from os.path import basename, join, isdir, isfile
import math
import numpy as np

def aggregate_frame_hues(hues, hue_width):
    aggr = {}
    for hue_idx in range(len(hues)):
        bin_idx = int(hue_idx/hue_width)
        if bin_idx not in aggr:
            aggr[bin_idx] = 0
        aggr[bin_idx] += hues[hue_idx]
    return aggr

def get_video_files(training_dir):
    files = []
    dirs = [join(training_dir, o) for o in listdir(training_dir) if isdir(join(training_dir, o))]
    for d in dirs:
        files += [join(d, f) for f in listdir(d) if isfile(join(d, f)) and f.endswith(".bin")]
    return files

def main(training_data, outdir, pf_cutoff, hue_width):
    cross_video_bin_scores = {}
    plot_suffix = "pf_cutoff_%s_hue_bin_%s"%(str(pf_cutoff), str(hue_width))
    median_hues = {}
    median_hues_aggr = {}
    sum_hues_aggr = {}
    colors = {True:"red", False:"blue"}
    video_files = get_video_files(training_data)
    num_vids = len(video_files)
    cols = 3
    rows = math.ceil(num_vids/float(cols))
    plt.close()
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(24,8))

    HUE_RANGE=256
    num_hue_bins = math.ceil(HUE_RANGE/float(hue_width))

    vid_idx = 0
    for video_file in video_files:
        row = int(vid_idx/cols)
        col = vid_idx - row*cols
        ax = axs[row][col]

        hue_aggr = {True:None, False:None}
        vid_name = basename(video_file)[:-4]
        absolute_pixel_count = False
        
        # Extracting the features from training data samples
        observations = python_server.mapping_features.read_samples(video_file)
        frame_id=0
        for observation in observations:
            if frame_id <= 1:
                frame_id += 1
                continue

            label = observation.label
            hues = observation.feats.feats[0].feat.wholeHisto.histo.counts[0].count
            total_pixels = observation.feats.feats[0].feat.wholeHisto.histo.totalCountedPixels

            # Aggregate the pixel counts into the Hue bins
            aggr_frame_hues = aggregate_frame_hues(hues, hue_width)            

            # Normalizing the pixel counts by total_pixels to get Pixel Fractions
            norm_hues = {}
            for hue_bin in aggr_frame_hues:
                norm_hues[hue_bin] = aggr_frame_hues[hue_bin]/float(total_pixels)
            
            if hue_aggr[label] == None:
                hue_aggr[label] = [[], []]

            for hue_bin in norm_hues:
                if norm_hues[hue_bin] == 0 or norm_hues[hue_bin] < pf_cutoff:
                    continue
                hue_aggr[label][0].append(hue_bin*hue_width)
                hue_aggr[label][1].append(norm_hues[hue_bin])

                if vid_name not in median_hues:
                    median_hues[vid_name] = {True: {}, False: {}}
                if hue_bin not in median_hues[vid_name][label]:
                    median_hues[vid_name][label][hue_bin] = []
                median_hues[vid_name][label][hue_bin].append(norm_hues[hue_bin])

        for label in [False, True]:
            if hue_aggr[label] == None: # or label== False:
                continue
            ax.scatter(hue_aggr[label][0], hue_aggr[label][1], marker=".", color=colors[label])

        ax.set_ylabel("Pixel fraction")
        ax.set_xlabel("Hue value")

        median_hues_aggr[vid_name] = {True: {}, False: {}}
        sum_hues_aggr[vid_name] = {True: {}, False: {}}
        for label in median_hues_aggr[vid_name]:
            for hue_bin in median_hues[vid_name][label]:
                med = 0
                if hue_bin not in median_hues[vid_name][label]:
                    continue
                if len(median_hues[vid_name][label][hue_bin]) > 0:
                    med = np.average(median_hues[vid_name][label][hue_bin])
                median_hues_aggr[vid_name][label][hue_bin] = med
                sum_hues_aggr[vid_name][label][hue_bin] = sum(median_hues[vid_name][label][hue_bin])

        vid_idx += 1
    fig.savefig(join(outdir, "pixel_fractions_%s.png"%plot_suffix), bbox_inches="tight")

    plt.close()
    vid_idx = 0
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(24,8))
    for vid in median_hues_aggr:
        row = int(vid_idx/cols)
        col = vid_idx - row*cols
        ax = axs[row][col]
    
        for label in [False, True]:
            X = []
            Y = []
       
            for hue_bin in median_hues_aggr[vid][label]:
                if median_hues_aggr[vid][label][hue_bin] == 0:
                    continue
                X.append(hue_bin*hue_width)
                Y.append(median_hues_aggr[vid][label][hue_bin])

            ax.scatter(X, Y, marker=".", color=colors[label])
            ax.set_ylabel("Median PF")
            ax.set_xlabel("Hue value")
        vid_idx += 1
    fig.savefig(join(outdir, "median_pixel_fractions_%s.png"%plot_suffix), bbox_inches="tight")

    plt.close()
    vid_idx = 0
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(24,8))
    for vid in sum_hues_aggr:
        row = int(vid_idx/cols)
        col = vid_idx - row*cols
        ax = axs[row][col]
    
        for label in [True]:
            X = []
            Y = []
       
            for hue_bin in median_hues_aggr[vid][label]:
                X.append(hue_bin*hue_width)
                Y.append(sum_hues_aggr[vid][label][hue_bin])

            max_count = max([0]+[len(median_hues[vid][label][hue_bin]) for hue_bin in median_hues[vid][label]])
            for hue_bin in sum_hues_aggr[vid][label]:
                if hue_bin not in cross_video_bin_scores:
                    cross_video_bin_scores[hue_bin] = 0
                if max_count > 0:
                    cross_video_bin_scores[hue_bin] += len(median_hues[vid][label][hue_bin])

            ax.bar(X, Y, color=colors[label], width=hue_width)
            ax.set_ylabel("Sum of PF")
            ax.set_xlabel("Hue value")
        vid_idx += 1
    fig.savefig(join(outdir, "sum_pixel_fractions_%s.png"%plot_suffix), bbox_inches="tight")

    plt.close()
    vid_idx = 0
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(24,8))
    for vid in median_hues_aggr:
        row = int(vid_idx/cols)
        col = vid_idx - row*cols
        ax = axs[row][col]
    
        for label in [True]:
            X = []
            Y = []
       
            for hue_bin in median_hues[vid][label]:
                X.append(hue_bin*hue_width)
                Y.append(len(median_hues[vid][label][hue_bin]))

            ax.bar(X, Y, color=colors[label], width=hue_width)
            #ax.scatter(X, Y, marker=".", color=colors[label])
            ax.set_ylabel("Count")
            ax.set_xlabel("Hue value")
            ax.set_xlim([0, 180])
        vid_idx += 1
    fig.savefig(join(outdir, "count_pixel_fractions_%s.png"%plot_suffix), bbox_inches="tight")

    plt.close()
    vid_idx = 0
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(24,8))
    for vid in median_hues_aggr:
        row = int(vid_idx/cols)
        col = vid_idx - row*cols
        ax = axs[row][col]
    
        for label in [True]:
            X = []
            Y = []
       
            for hue_bin in median_hues_aggr[vid][label]:
                X.append(hue_bin*hue_width)
                Y.append(len(median_hues[vid][label][hue_bin])*median_hues_aggr[vid][label][hue_bin])

            ax.bar(X, Y, color=colors[label], width=hue_width)
            #ax.scatter(X, Y, marker=".", color=colors[label])
            ax.set_ylabel("Count")
            ax.set_xlabel("Hue value")
            ax.set_xlim([0, 180])
        vid_idx += 1
    fig.savefig(join(outdir, "median_times_count_pixel_fractions_%s.png"%plot_suffix), bbox_inches="tight")

    plt.close()
    scores = [(x, cross_video_bin_scores[x]) for x in cross_video_bin_scores]
    scores = sorted(scores, key = lambda x : x[0])
    X = [x[0]*hue_width for x in scores]
    Y = [x[1] for x in scores]
    plt.bar(X, Y, width=hue_width, edgecolor="black")
    plt.xlabel("Hue")
    plt.ylabel("Score of each Hue bin for given query")
    plt.xticks(X)
    plt.savefig(join(outdir, "cross_video_scores_%s.png"%plot_suffix), bbox_inches="tight")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzing the distribution of Pixel Fraction")
    parser.add_argument("-D", "--training-data", dest="training_data", help="Path to training data")
    parser.add_argument("-O", "--out-dir", dest="output_dir", help="Path to output data that would contain plots")
    parser.add_argument("--cutoff", dest="pf_cutoff", help="Cutoff of Pixel Fraction", type=float)
    parser.add_argument("--hue-width", dest="hue_width", help="Width of each hue bin. Default=5", default=5, type=float)
    
    args = parser.parse_args()
    main(args.training_data, args.output_dir, args.pf_cutoff, args.hue_width)
