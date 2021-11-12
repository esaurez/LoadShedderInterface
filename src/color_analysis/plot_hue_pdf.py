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
import numpy as np

def main(training_data, outdir, pf_cutoff):
    median_hues = {}
    median_hues_aggr = {}
    colors = {True:"red", False:"blue"}
    video_files = glob.glob(training_data+"/*.bin")
    num_vids = len(video_files)
    cols = 3
    rows = math.ceil(num_vids/float(cols))
    plt.close()
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(24,8))

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
            if frame_id <= 5:
                frame_id += 1
                continue

            label = observation.label
            hues = observation.feats.feats[0].feat.wholeHisto.histo.counts[0].count
            total_pixels = observation.feats.feats[0].feat.wholeHisto.histo.totalCountedPixels

            norm_hues = [o/float(total_pixels) for o in hues]
            
            if hue_aggr[label] == None:
                hue_aggr[label] = [[], []]

            for idx in range(len(hues)):
                if norm_hues[idx] == 0 or norm_hues[idx] < pf_cutoff:
                    continue
                hue_aggr[label][0].append(idx)
                hue_aggr[label][1].append(norm_hues[idx])

                if vid_name not in median_hues:
                    median_hues[vid_name] = {True: [[] for x in range(256)], False:[[] for x in range(256)]}
                median_hues[vid_name][label][idx].append(norm_hues[idx])


        for label in [False, True]:
            if hue_aggr[label] == None: # or label== False:
                continue
            ax.scatter(hue_aggr[label][0], hue_aggr[label][1], marker=".", color=colors[label])

        ax.set_ylabel("Pixel fraction")
        ax.set_xlabel("Hue value")

        median_hues_aggr[vid_name] = {True: [], False: []}
        for label in median_hues_aggr[vid_name]:
            for idx in range(len(median_hues[vid_name][label])):
                med = 0
                if len(median_hues[vid_name][label][idx]) > 0:
                    med = np.median(median_hues[vid_name][label][idx])
                median_hues_aggr[vid_name][label].append(med)

        vid_idx += 1
    fig.savefig(join(outdir, "pixel_fractions_pf_cutoff_%.2f.png"%pf_cutoff), bbox_inches="tight")

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
       
            for idx in range(len(median_hues_aggr[vid][label])):
                if median_hues_aggr[vid][label][idx] == 0:
                    continue
                X.append(idx)
                Y.append(median_hues_aggr[vid][label][idx])

            ax.scatter(X, Y, marker=".", color=colors[label])
            ax.set_ylabel("Median PF")
            ax.set_xlabel("Hue value")
        vid_idx += 1
    fig.savefig(join(outdir, "median_pixel_fractions_pf_cutoff_%.2f.png"%pf_cutoff), bbox_inches="tight")

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
       
            for idx in range(len(median_hues_aggr[vid][label])):
                X.append(idx)
                Y.append(len(median_hues[vid][label][idx]))

            ax.bar(X, Y, color=colors[label])
            #ax.scatter(X, Y, marker=".", color=colors[label])
            ax.set_ylabel("Count")
            ax.set_xlabel("Hue value")
            ax.set_xlim([0, 180])
        vid_idx += 1
    fig.savefig(join(outdir, "count_pixel_fractions_pf_cutoff_%.2f.png"%pf_cutoff), bbox_inches="tight")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzing the distribution of Pixel Fraction")
    parser.add_argument("-D", "--training-data", dest="training_data", help="Path to training data")
    parser.add_argument("-O", "--out-dir", dest="output_dir", help="Path to output data that would contain plots")
    parser.add_argument("--cutoff", dest="pf_cutoff", help="Cutoff of Pixel Fraction", type=float)
    
    args = parser.parse_args()
    main(args.training_data, args.output_dir, args.pf_cutoff)
