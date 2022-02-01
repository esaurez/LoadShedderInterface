import argparse
from os import listdir
from os.path import join, isfile, isdir
import yaml
import pandas as pd
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../color_analysis"))
sys.path.insert(0, os.path.join(script_dir, "../color_analysis/sv_matrix"))
import object_based_metrics_calc
from random_shedding import *

def compute_util_threshold(global_cdf, drop_rate):
    for idx in range(len(global_cdf)):
        (ratio, util) = global_cdf[idx]
        if ratio > drop_rate:
            prev = global_cdf[idx-1]
            slope = (util - prev[1])/(ratio - prev[0])

            util_threshold = prev[1] + slope*(drop_rate - prev[0])
            return util_threshold
        elif drop_rate == ratio:
            return util

def main(training_conf):
    conf_file = join(training_conf, "conf.yaml")
    with open(conf_file) as f:
        conf = yaml.safe_load(f)
    training_dir = conf["training_dir"]
    training_split = conf["training_split"]

    vids = [v for v in listdir(training_dir) if isdir(join(training_dir, v))]

    cdfs = {}
    per_vid_frames = {}
    dropped = {}
    uniq_objs = {}

    for vid in vids:
        vid_dir = join(training_dir, vid)
        utils_df = pd.read_csv(join(vid_dir, "frame_utils.csv"))

        # Creating the list of test frames for this video
        utils = [row["utility"] for idx, row in utils_df.iterrows()]
        labels = [row["label"] for idx, row in utils_df.iterrows()]
        num_training_frames = int(training_split*len(utils))
        per_vid_frames[vid] = []
        for idx in range(num_training_frames, len(utils)):
            per_vid_frames[vid].append((idx, utils[idx], labels[idx]))

        # Saving the CDF of the video
        cdf = []
        cdf_file = join(vid_dir, "util_cdf.txt")
        with open(cdf_file) as f:
            for line in f.readlines():
                s = line.split()
                cdf.append((float(s[0]), float(s[1])))
        cdfs[vid] = cdf
        uniq_objs[vid] = object_based_metrics_calc.get_obj_frames(join(vid_dir, "unique_objs_per_frame.txt"))

    # Now combine videos from diff cameras to make a stream
    stream = []
    idxs = {}
    for vid in vids:
        idxs[vid] = 0
    go = True
    while go:
        go = False
        for vid in vids:
            curr_idx = idxs[vid]
            if curr_idx >= len(per_vid_frames[vid]):
                continue
            (frame_idx, util, label) = per_vid_frames[vid][curr_idx]
            stream.append((vid, frame_idx, util, label))
            idxs[vid] += 1
            go = True

    drop_ratios = []
    d = 0
    while d < 1.0:
        drop_ratios.append(d)
        d += 0.025

    num_trials = 10
    random_X = []
    random_Y = []
    random_frame_drops = []

    utils_X = []
    utils_Y = []
    utils_frame_drops = []

    for drop_ratio in drop_ratios:
        util_thresholds = {} # per video
        for vid in vids:
            util_thresholds[vid] = compute_util_threshold(cdfs[vid], drop_ratio)

        for utils_approach in [True, False]:
            if utils_approach:
                num_trials = 1
            else:
                num_trials = 10

            for trial in range(num_trials):
                total_objs = 0
                objs_detected = 0
                total_frames = 0
                frames_dropped = 0

                for vid in vids:
                    dropped[vid] = [False for idx in range(per_vid_frames[vid][-1][0]+1)]

                if utils_approach:
                    for (vid, frame_idx, util, label) in stream:
                        util_threshold = util_thresholds[vid]
                        dropped[vid][frame_idx] = util <= util_threshold
                    
                else:
                    random.seed(str(drop_ratio)+"_"+str(trial)) 
    
                    for (vid, frame_idx, util, label) in stream:
                        r = random.random()
                        drop = r < drop_ratio
                        dropped[vid][frame_idx] = drop

                for vid in vids:
                    num_training_frames = int(training_split*len(dropped[vid]))
                    obj_covered = object_based_metrics_calc.get_obj_coverage(uniq_objs[vid], dropped[vid], num_training_frames)

                    total_objs += len(obj_covered)
                    objs_detected += len([x for x in obj_covered if obj_covered[x]])

                    total_frames += len(dropped[vid]) - num_training_frames
                    frames_dropped += len([x for x in dropped[vid][num_training_frames:] if x])

                obj_det_rate = objs_detected/float(total_objs)
                frame_drop_rate = frames_dropped/float(total_frames)

                if utils_approach:
                    utils_X.append(drop_ratio)
                    utils_Y.append(obj_det_rate)
                    utils_frame_drops.append(frame_drop_rate)
                else:
                    random_X.append(drop_ratio)
                    random_Y.append(obj_det_rate)
                    random_frame_drops.append(frame_drop_rate)
    
    plt.close()
    fontsize=20
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(random_X, random_Y, label="Random")
    ax.scatter(utils_X, utils_Y, label="Utility-based")
    fontP = FontProperties()
    fontP.set_size(fontsize)
    ax.legend(prop=fontP)
    ax.set_xlabel("Target drop ratio", fontsize=fontsize)
    ax.set_ylabel("Fraction of target objects detected", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    fig.savefig("merged_streams_obj_rate.png", bbox_inches="tight")

    plt.close()
    fontsize=20
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(random_X, random_frame_drops, label="Random")
    ax.scatter(utils_X, utils_frame_drops, label="Utility-based")
    fontP = FontProperties()
    fontP.set_size(fontsize)
    ax.legend(prop=fontP)
    ax.set_xlabel("Target drop ratio", fontsize=fontsize)
    ax.set_ylabel("Observed drop ratio", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    fig.savefig("merged_streams_observed_drop.png", bbox_inches="tight")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", dest="training_conf", help="Path to training conf dir", required=True)
    args = parser.parse_args()
    main(args.training_conf)
