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

def compute_object_metrics(training_dir, utils_df, mats_dir, util_threshold):
    aggr_total_objs = 0
    aggr_detected_objs = 0

    total_frames = 0
    frames_dropped = 0

    grouping = utils_df.groupby(["vid_name", "cv_fold"])
    for group in grouping.groups.keys():
        (vid, fold) = group
        gdf = grouping.get_group(group)

        uniq_obj_file = join(join(training_dir, vid), "unique_objs_per_frame.txt")
        if not isfile(uniq_obj_file):
            continue
        obj_frames = object_based_metrics_calc.get_obj_frames(uniq_obj_file)

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
    cv_folds = 3
    num_test_vids = int(len(vids)/cv_folds)
    for cv_fold in range(cv_folds):
        test_vids = [vids[idx] for idx in range(len(vids)) if idx >= cv_fold*num_test_vids and idx < (cv_fold+1)*num_test_vids]
        train_vids = [vids[idx] for idx in range(len(vids)) if idx < cv_fold*num_test_vids or idx >= (cv_fold+1)*num_test_vids]

        # Aggregate sv_matrices in training_set
        
        ## Seperate out the positive and negative matrices
        positive_mats = []
        negative_mats = []
        for train_vid in train_vids:
            vid_dir = join(training_dir, train_vid)
            positive_mats.append(join(vid_dir, "sv_matrix_label_True_BINS_%d_C_%s.txt"%(num_bins, color)))
            negative_mats.append(join(vid_dir, "sv_matrix_label_False_BINS_%d_C_%s.txt"%(num_bins, color)))

        aggr_mat, max_val = aggregate_sv_matrix.aggregate(num_bins, positive_mats, negative_mats)

        # Now compute the utility for test frames
        for test_vid in test_vids:
            test_vid_dir = join(training_dir, test_vid)
            bin_file = [join(test_vid_dir
, f) for f in listdir(test_vid_dir) if isfile(join(test_vid_dir, f)) and f.endswith(".bin")][0]
            ground_truth_frames = python_server.mapping_features.read_samples(bin_file)
            mats = read_per_frame_mats(join(mats_dir, test_vid))
            frame_idx = 0
            for mat in mats:
                label = ground_truth_frames[frame_idx].label
                count = ground_truth_frames[frame_idx].detections.totalDetections
                util  = compute_test_frames_util.compute_util(mat, aggr_mat[True])
                raw_data.append([count, label, frame_idx, util, test_vid, cv_fold])
                frame_idx += 1

    df = pd.DataFrame(raw_data, columns=["count", "label", "frame_idx", "utility", "vid_name", "cv_fold"])
    df.to_csv(join(mats_dir, "frame_utils.csv"))

    plt.close()
    fig, ax = plt.subplots(figsize=(24,8))
    sns.boxplot(data=df, x="vid_name", y="utility", hue="label", ax=ax)
    ax.set_xlabel("Label of the frame")
    ax.set_ylabel("Utility for %s"%color)
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
    fig.savefig(join(mats_dir, "combined_util_CROSSVIDEO.png"), bbox_inches="tight")

    thresholds = []
    obj_rates = []
    frame_rates = []
    for util_threshold in np.arange(0, 0.04, 0.001):
        obj_rate, frame_rate = compute_object_metrics(training_dir, df, mats_dir, util_threshold)
        obj_rates.append(obj_rate)
        frame_rates.append(frame_rate)
        thresholds.append(util_threshold)

    plt.close()
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(thresholds, obj_rates, color="red")
    ax.set_xlabel("Utility threshold")
    ax.set_ylabel("Fraction of objects detected")
    ax2 = ax.twinx()
    ax2.plot(thresholds, frame_rates, color="blue")
    ax2.set_ylabel("Fraction of frames dropped")
    fig.savefig(join(mats_dir, "object_based_drops.png"), bbox_inches="tight")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", dest="training_conf", help="Directory containing training conf")
    parser.add_argument("-M", dest="mats_dir", help="Directory containing per frame matrix")
    args = parser.parse_args()

    main(args.training_conf, args.mats_dir)
