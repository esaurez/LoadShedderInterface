import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../../"))
sys.path.insert(0, os.path.join(script_dir, "../"))
import python_server.mapping_features
import object_based_metrics_calc
import yaml
from os.path import isfile, join, isdir
from os import listdir
import pandas as pd
import aggregate_sv_matrix
import compute_test_frames_util

def read_per_frame_mats(folder):
    files = [f for f in listdir(folder) if isfile(join(folder, f)) and f.startswith("frame_") and f.endswith(".txt")]
    files = sorted(files, key = lambda x: int(x[:-4].split("_")[1]))
    mats = []
    for matfile in files:
        mat =[]
        with open(join(folder, matfile)) as f:
            for line in f.readlines():
                mat.append([])
                s = line.split()
                for val in s:
                    mat[-1].append(float(val))
        mats.append(mat)
    return mats

def compute_object_metrics(training_dir, utils_df, util_threshold, num_training_frames, vid_obj_frames):
    aggr_total_objs = 0
    aggr_detected_objs = 0

    total_frames = 0
    frames_dropped = 0

    grouping = utils_df.groupby("vid_name")
    for group in grouping.groups.keys():
        vid = group
        gdf = grouping.get_group(group)

        if vid not in vid_obj_frames:
            continue
        obj_frames = vid_obj_frames[vid]

        # Getting the utilities for each frame
        frames = []
        for idx, row in gdf.iterrows():
            frame_idx = row["frame_idx"]
            util = row["utility"]
            frames.append((frame_idx, util))
        frames = sorted(frames, key = lambda x : x[0])
        utils = [x[1] for x in frames]

        total_frames += len(utils)
        frames_dropped += len([f for f in utils if f < util_threshold])

        obj_covered = {}
        for obj in obj_frames:
            obj_in_training_data = False
            obj_found = False
            frames = obj_frames[obj]

            for fidx in frames:
                if fidx < num_training_frames[vid]:
                    obj_in_training_data = True
                if utils[fidx] >= util_threshold:
                    obj_found = True

            if not obj_in_training_data:
                global_obj_id = vid+str(obj)
                obj_covered[global_obj_id] = obj_found


        if len(obj_covered) != 0:
            object_detection_rate = len([x for x in obj_covered if obj_covered[x] == True])/float(len(obj_covered))

            aggr_total_objs += len(obj_covered)
            aggr_detected_objs += len([x for x in obj_covered if obj_covered[x] == True])

    frame_drop_rate = frames_dropped/float(total_frames) 
    return float(aggr_detected_objs)/aggr_total_objs, frame_drop_rate        

def main(training_conf_dir, mats_dir):
    # Read the conf
    with open(join(training_conf_dir, "conf.yaml")) as fi:
        conf = yaml.safe_load(fi)
    training_dir = conf["training_dir"]
    training_split = conf["training_split"]
    num_bins = conf["num_bins"]
    if "is_composite" in conf and conf["is_composite"] == "true":
        print ("Composite query not supported")
        exit(0)

    # Extract color from the conf
    for hue_bin in conf["hue_bins"]:
        color = hue_bin["name"]
        break

    raw_data = []

    # List the videos
    vids = [d for d in listdir(training_dir) if isdir(join(training_dir, d))]
    num_training_frames = {}

    for vid in vids:
        for pervideo in [True, False]:
            ## Seperate out the positive and negative matrices
            positive_mats = []
            negative_mats = []
            if pervideo:
                train_vids = [vid]
            else:
                train_vids = vids

            for train_vid in train_vids:
                vid_dir = join(training_dir, train_vid)
                positive_mats.append(join(vid_dir, "sv_matrix_label_True_BINS_%d_C_%s.txt"%(num_bins, color)))
                negative_mats.append(join(vid_dir, "sv_matrix_label_False_BINS_%d_C_%s.txt"%(num_bins, color)))

            aggr_mat, max_val = aggregate_sv_matrix.aggregate(num_bins, positive_mats, negative_mats)

            # Now compute the utility for test frames
            test_vids = [vid]
            for test_vid in test_vids:
                test_vid_dir = join(training_dir, test_vid)
                bin_file = [join(test_vid_dir
, f) for f in listdir(test_vid_dir) if isfile(join(test_vid_dir, f)) and f.endswith(".bin")][0]
                ground_truth_frames = python_server.mapping_features.read_samples(bin_file)
                num_training_frames[test_vid] = int(training_split*len(ground_truth_frames))
                mats = read_per_frame_mats(join(mats_dir, test_vid))
                frame_idx = 0
                for mat in mats:
                    label = ground_truth_frames[frame_idx].label
                    count = ground_truth_frames[frame_idx].detections.totalDetections
                    util  = compute_test_frames_util.compute_util(mat, aggr_mat[True])
                    raw_data.append([count, label, frame_idx, util, test_vid, pervideo, num_training_frames[test_vid]])
                    frame_idx += 1

    df = pd.DataFrame(raw_data, columns=["count", "label", "frame_idx", "utility", "vid_name", "pervideo", "num_training_frames"])
    df.to_csv(join(mats_dir, "frame_utils.csv"))

    def pervideo_label(x):
        if x:
            return "Local"
        else:
            return "Global"

    vid_grouping = df.groupby("vid_name")
    for vid_name in vid_grouping.groups.keys():
        gdf = vid_grouping.get_group(vid_name)
        gdf = gdf[gdf["frame_idx"]<gdf["num_training_frames"]]
        gdf["pervideo"] = gdf["pervideo"].apply(lambda x: pervideo_label(x))
        plt.close()
        fig, ax = plt.subplots(figsize=(6,8))
        sns.boxplot(data=gdf, x="pervideo", y="utility", hue="label", ax=ax)
        ax.set_xlabel("Model Type")
        ax.set_ylabel("Utility for %s"%color)
        fig.savefig(join(mats_dir, "local_vs_global_%s.png"%vid_name), bbox_inches="tight")

    vid_objs = {}
    for vid in vids:
        uniq_obj_file = join(join(training_dir, vid), "unique_objs_per_frame.txt")
        if not isfile(uniq_obj_file):
            continue
        vid_objs[vid] = object_based_metrics_calc.get_obj_frames(uniq_obj_file)

    thresholds = {True: [], False: []}
    obj_rates = {True: [], False: []}
    frame_rates = {True: [], False: []}
    pervid_grouping = df.groupby("pervideo")
    for pervid in pervid_grouping.groups.keys():
        gdf = pervid_grouping.get_group(pervid)
        for util_threshold in np.arange(0, 0.04, 0.001):
        #for util_threshold in np.arange(0, 0.04, 0.0005):
            obj_rate, frame_rate = compute_object_metrics(training_dir, gdf, util_threshold, num_training_frames, vid_objs)
            obj_rates[pervid].append(obj_rate)
            frame_rates[pervid].append(frame_rate)
            thresholds[pervid].append(util_threshold)

    linestyles={True:"-", False:":"}

    plt.close()
    fig, ax = plt.subplots(figsize=(8,6))
    ax2 = ax.twinx()
    for pervid in [True, False]:
        ls = linestyles[pervid]
        ax.plot(thresholds[pervid], obj_rates[pervid], color="red", linestyle=ls, label=pervideo_label(pervid))
        ax2.plot(thresholds[pervid], frame_rates[pervid], color="blue", linestyle=ls)
    ax.set_xlabel("Utility threshold")
    ax.set_ylabel("Fraction of objects detected")
    ax2.set_ylabel("Fraction of frames dropped")
    ax.legend()
    fig.savefig(join(mats_dir, "object_based_drops.png"), bbox_inches="tight")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", dest="training_conf", help="Directory containing training conf")
    parser.add_argument("-M", dest="mats_dir", help="Directory containing per frame matrix")
    args = parser.parse_args()

    main(args.training_conf, args.mats_dir)
