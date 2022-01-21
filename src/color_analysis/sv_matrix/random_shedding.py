import argparse
import yaml
from os import listdir
from os.path import join, isdir, isfile
import os, sys
import random
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../../"))
sys.path.insert(0, os.path.join(script_dir, "../"))
import python_server.mapping_features
import object_based_metrics_calc

def main(training_conf, outdir, frame_utils):
    num_trials = 20
    conf_file = join(training_conf, "conf.yaml")
    with open(conf_file) as f:
        conf = yaml.safe_load(f)
    training_dir = conf["training_dir"]

    drop_ratios = []
    d = 0
    while d < 1.0:
        drop_ratios.append(d)
        d += 0.025

    vids = [d for d in listdir(training_dir) if isdir(join(training_dir, d))]
    obj_frames = {}
    for vid in vids:
        vid_dir= join(training_dir, vid)
        bin_file = [join(vid_dir, f) for f in listdir(vid_dir) if isfile(join(vid_dir, f)) and f.endswith(".bin")][0]
        frames = python_server.mapping_features.read_samples(bin_file)

        uniq_obj = join(vid_dir,  "unique_objs_per_frame.txt")
        if not isfile(uniq_obj):
            continue
        obj_frames[vid] = object_based_metrics_calc.get_obj_frames(uniq_obj)
    
    # Read the frame_utils.csv
    df = pd.read_csv(frame_utils)
    utils_drop_rates = {}
    utils_obj_rates = {}
    utils_target_drop_rates = {}
    for drop_rate in drop_ratios:
    #for util_threshold in np.arange(0, 0.04, 0.001):
        grouping = df.groupby("cv_fold")
        for group in grouping.groups.keys():
            if group not in utils_drop_rates:
                utils_drop_rates[group] = []
                utils_obj_rates[group] = []
                utils_target_drop_rates[group] = []
            total_frames = 0
            frames_dropped = 0
            total_objs = 0
            objs_detected = 0
            cv_fold_gdf = grouping.get_group(group)
            vid_group = cv_fold_gdf.groupby("vid_name")
            for vid in vid_group.groups.keys():
                gdf = vid_group.get_group(vid)
                utils = []
                for idx, row in gdf.iterrows():
                    utils.append(row["utility"])
                # Calculate the utility threshold based on the drop rate
                sorted_utils = sorted(utils)
                util_threshold = sorted_utils[min(int(len(utils)*drop_rate), len(utils)-1)]
                is_frame_dropped = [u <= util_threshold for u in utils]
                if vid not in obj_frames:
                    continue
                obj_covered = object_based_metrics_calc.get_obj_coverage(obj_frames[vid], is_frame_dropped, 0)
                frames_dropped += len([u for u in utils if u <= util_threshold])
                total_frames += len(utils)
                total_objs += len(obj_covered)
                objs_detected += len([x for x in obj_covered if obj_covered[x]])
            utils_drop_rates[group].append(frames_dropped/float(total_frames))
            utils_obj_rates[group].append(objs_detected/float(total_objs))
            utils_target_drop_rates[group].append(drop_rate)

    #twin_plot_data = []
    #twin_plot_data.append((utils_drop_rates, utils_obj_rates, utils_target_drop_rates))

    # Calculating for random
    rand_utils_drop_rates = {}
    rand_utils_obj_rates = {}
    rand_utils_target_drop_rates = {}
    for drop_rate in drop_ratios:
        for trial in range(num_trials):
            #for util_threshold in np.arange(0, 0.04, 0.001):
            grouping = df.groupby("cv_fold")
            for group in grouping.groups.keys():
                if group not in rand_utils_drop_rates:
                    rand_utils_drop_rates[group] = []
                    rand_utils_obj_rates[group] = []
                    rand_utils_target_drop_rates[group] = []
                total_frames = 0
                frames_dropped = 0
                total_objs = 0
                objs_detected = 0
                cv_fold_gdf = grouping.get_group(group)
                vid_group = cv_fold_gdf.groupby("vid_name")
                for vid in vid_group.groups.keys():
                    gdf = vid_group.get_group(vid)
                    utils = []
                    if vid not in obj_frames:
                        continue
                    random.seed(vid+str(trial))
                    R = []
                    for idx in range(len(gdf)):
                        R.append(random.random())
                    
                    is_frame_dropped = []
                    for idx in range(len(R)):
                        drop = R[idx] < drop_rate
                        is_frame_dropped.append(drop)
                    obj_covered = object_based_metrics_calc.get_obj_coverage(obj_frames[vid], is_frame_dropped, 0)
                    frames_dropped += len([u for u in is_frame_dropped if u])
                    total_frames += len(is_frame_dropped)
                    total_objs += len(obj_covered)
                    objs_detected += len([x for x in obj_covered if obj_covered[x]])
                rand_utils_drop_rates[group].append(frames_dropped/float(total_frames))
                rand_utils_obj_rates[group].append(objs_detected/float(total_objs))
                rand_utils_target_drop_rates[group].append(drop_rate)

    
    Y = []
    for target_drop_ratio in drop_ratios:
        Y.append([])
        for trial in range(num_trials):
            total_objs = 0
            total_detected_objs = 0
            for vid in vids:
                if vid not in obj_frames:
                    continue
                random.seed(vid+str(trial))
                R = []
                for idx in range(len(frames)):
                    R.append(random.random())
                
                is_frame_dropped = []
                for idx in range(len(R)):
                    drop = R[idx] < target_drop_ratio
                    is_frame_dropped.append(drop)
                obj_covered = object_based_metrics_calc.get_obj_coverage(obj_frames[vid], is_frame_dropped, 0)
                total_objs += len(obj_covered)
                total_detected_objs += len([x for x in obj_covered if obj_covered[x]])
            obj_detection_rate = float(total_detected_objs)/total_objs
            Y[-1].append(obj_detection_rate)

    random_drop_rates = []
    random_obj_rates = []
    for idx in range(len(drop_ratios)):
        drop_rate = drop_ratios[idx]
        for r in Y[idx]:
            random_drop_rates.append(drop_rate)
            random_obj_rates.append(r)
    
    #twin_plot_data.append((random_drop_rates, random_obj_rates, random_drop_rates))

    # Plotting with target drop rate as the control variable
    labels = ["Utility-based", "Random"]
    for group in utils_drop_rates:
        plt.close()
        fig, ax = plt.subplots(figsize=(8,6))
        ax.scatter(utils_target_drop_rates[group], utils_obj_rates[group], color="blue")
        ax2 = ax.twinx()
        ax2.scatter(utils_target_drop_rates[group], utils_drop_rates[group], color="red")
        ax.set_xlabel("Target drop rate")
        ax.set_ylabel("Fraction of target objects detected")
        ax2.set_ylabel("Frame drop rate")
        ax.set_ylim([0,1.1])
        ax2.set_ylim([0,1.1])
        ax.set_title("utility_%d"%group)
        fig.savefig(join(outdir, "rates_vs_target_drop_rate_group_%d.png"%group))

    for group in rand_utils_drop_rates:
        plt.close()
        fig, ax = plt.subplots(figsize=(8,6))
        ax.scatter(rand_utils_target_drop_rates[group], rand_utils_obj_rates[group], color="blue")
        ax2 = ax.twinx()
        ax2.scatter(rand_utils_target_drop_rates[group], rand_utils_drop_rates[group], color="red")
        ax.set_xlabel("Target drop rate")
        ax.set_ylabel("Fraction of target objects detected")
        ax2.set_ylabel("Frame drop rate")
        ax.set_ylim([0,1.1])
        ax2.set_ylim([0,1.1])
        ax.set_title("random_%s"%group)
        fig.savefig(join(outdir, "random_rates_vs_target_drop_rate_group_%d.png"%group))

    aggr_util_drop_rates = []
    aggr_util_obj_rates = []
    for group in utils_drop_rates:
        for idx in range(len(utils_drop_rates[group])):
            aggr_util_drop_rates.append(utils_drop_rates[group][idx])
            aggr_util_obj_rates.append(utils_obj_rates[group][idx])
    plt.close()
    plt.scatter(random_drop_rates, random_obj_rates, label="Random")
    plt.scatter(aggr_util_drop_rates, aggr_util_obj_rates, label="Utility-based")
    plt.legend()
    plt.xlabel("Observed drop rate of frames")
    plt.ylabel("Fraction of target objects detected")
    plt.savefig(join(outdir, "random_comparison.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", dest="training_conf", help="Path to training conf dir", required=True)
    parser.add_argument("-U", dest="frame_utils", help="Path to frame_utils.csv calculated using utility based method", required=True)
    parser.add_argument("-O", dest="outdir", help="Path to output directory", default="/tmp")
    
    args = parser.parse_args()

    main(args.training_conf, args.outdir, args.frame_utils)
