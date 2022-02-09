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
from matplotlib.font_manager import FontProperties

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../../"))
sys.path.insert(0, os.path.join(script_dir, "../"))
import python_server.mapping_features
import object_based_metrics_calc

class FrameDropCondition:
    def compute_is_frame_dropped(utils, vid):
        return None

class UtilBasedFrameDropCondition:
    def __init__(self, target_ratio):
        self.target_ratio = target_ratio

    def get_frame_drop_bools(self, utils, cdf_ratio):
        num_cdf_frames = int(cdf_ratio*len(utils))
        sorted_utils = sorted(utils[:num_cdf_frames])
        util_threshold = sorted_utils[min(int(num_cdf_frames*self.target_ratio), num_cdf_frames-1)]
        is_frame_dropped = [u <= util_threshold for u in utils]        
        return is_frame_dropped

class RandomFrameDropCondition:
    def __init__(self, target_ratio):
        self.target_ratio  = target_ratio
        random.seed(self)

    def get_frame_drop_bools(self, utils, cdf_ratio):
        R = []
        for idx in range(len(utils)):
            R.append(random.random())
        
        is_frame_dropped = []
        for idx in range(len(R)):
            drop = R[idx] < self.target_ratio
            is_frame_dropped.append(drop)
        return is_frame_dropped

def compute_drop_obj_metrics(cv_fold, cv_fold_gdf, utils_data, frame_drop_condition, obj_frames, target_drop_rate, cdf_ratio):
    total_frames = 0
    frames_dropped = 0
    total_objs = 0
    objs_detected = 0
    obj_sel_rates = []
    vids = cv_fold_gdf["vid_name"].unique()
    for vid in vids:
        utils = utils_data[(cv_fold, vid)]
        # Calculate the utility threshold based on the drop rate
        is_frame_dropped = frame_drop_condition.get_frame_drop_bools(utils, cdf_ratio)
        if vid not in obj_frames:
            continue
        num_cdf_frames = int(cdf_ratio*len(utils))
        obj_covered = object_based_metrics_calc.get_obj_coverage(obj_frames[vid], is_frame_dropped, num_cdf_frames)
        obj_frame_sel_rates = object_based_metrics_calc.get_obj_frame_sel_rates(obj_frames[vid], is_frame_dropped, num_cdf_frames)
        frames_dropped += len([idx for idx in range(len(is_frame_dropped)) if is_frame_dropped[idx] and idx >= num_cdf_frames])
        total_frames += len(utils)-num_cdf_frames
        total_objs += len(obj_covered)
        objs_detected += len([x for x in obj_covered if obj_covered[x]])
        obj_sel_rates += obj_frame_sel_rates
    obs_frame_drop_rate = frames_dropped/float(total_frames)
    obj_det_rate = objs_detected/float(total_objs)
    return (obs_frame_drop_rate, obj_det_rate, np.mean(obj_sel_rates))

def get_obj_frames(training_dir):
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
    return obj_frames

def main(training_conf, outdir, frame_utils, cdf_ratio):
    conf_file = join(training_conf, "conf.yaml")
    with open(conf_file) as f:
        conf = yaml.safe_load(f)
    training_dir = conf["training_dir"]

    drop_ratios = []
    d = 0
    while d < 1.0:
        drop_ratios.append(d)
        d += 0.025

    obj_frames = get_obj_frames(training_dir)

    # Read the frame_utils.csv
    df = pd.read_csv(frame_utils)
    # For each approach, we have 1 dict for target_drop_rates, obj_det_rates, obs_drop_rates
    data = [[{}, {}, {}, {}], [{},{},{},{}]]

    # First create a map from (cv_fold, vid) --> [utils array]
    grouping = df.groupby(["cv_fold" , "vid_name"])
    utils_data = {}
    for group in grouping.groups.keys():
        (fold, vid) = group
        gdf = grouping.get_group(group)
        utils = []
        for idx, row in gdf.iterrows():
            utils.append(row["utility"])
        utils_data[group] = utils

    grouping = df.groupby("cv_fold")
    for group in grouping.groups.keys():
        for approach in data:
            for d in approach:
                d[group] = []

        fold_df = grouping.get_group(group) 

        for drop_rate in drop_ratios:
            drop_conds = [RandomFrameDropCondition(drop_rate), UtilBasedFrameDropCondition(drop_rate)]
            for approach_idx in range(2):
                if approach_idx == 0: # Random
                    num_trials = 10
                else: # Util based
                    num_trials = 1

                for trial in range(num_trials):
                    if approach_idx == 0:
                        drop_cond = RandomFrameDropCondition(drop_rate)
                    else:
                        drop_cond = UtilBasedFrameDropCondition(drop_rate)

                    result = data[approach_idx]
                    
                    obs_frame_drop_rate, obj_det_rate, obj_frame_sel_rate = compute_drop_obj_metrics(group, fold_df, utils_data, drop_cond, obj_frames, drop_rate, cdf_ratio)

                    result[3][group].append(obj_frame_sel_rate)
                    result[2][group].append(obs_frame_drop_rate)
                    result[1][group].append(obj_det_rate)
                    result[0][group].append(drop_rate)

    fontsize=20

    # Plotting with target drop rate as the control variable
    labels = ["Content-agnostic", "Utility-based"]
    for result_idx in range(len(labels)):
        result = data[result_idx]
        for group in result[0]:
            plt.close()
            fig, ax = plt.subplots(figsize=(8,6))
            ax.scatter(result[0][group], result[3][group], color="blue")
            ax2 = ax.twinx()
            ax2.scatter(result[0][group], result[2][group], color="red")
            ax.set_xlabel("Target drop rate", fontsize=fontsize)
            ax.set_ylabel("Object-based QoR", fontsize=fontsize)
            ax2.set_ylabel("Frame drop rate", fontsize=fontsize)
            ax.set_ylim([0,1.1])
            ax2.set_ylim([0,1.1])
            ax.tick_params(axis='both', which='major', labelsize=fontsize)
            ax2.tick_params(axis='both', which='major', labelsize=fontsize)
            #ax.set_title("%s_%d"%(labels[result_idx], group))
            fig.savefig(join(outdir, "%s_rates_vs_target_drop_rate_R_%.1f__group_%d.png"%(labels[result_idx], cdf_ratio, group)), bbox_inches="tight")

    plt.close()
    fig, ax = plt.subplots(figsize=(8,6))
    for idx in range(len(labels)):
        label = labels[idx]
        result = data[idx]
        obs_drops = []
        obj_dets = []
        for group in result[0]:
            obs_drops += result[2][group]
            obj_dets += result[3][group]
        ax.scatter(obs_drops, obj_dets, label=label)
    fontP = FontProperties()
    fontP.set_size(fontsize)
    ax.legend(prop=fontP)
    ax.set_xlabel("Observed drop rate of frames", fontsize=fontsize)
    ax.set_ylabel("Object-based QoR", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    fig.savefig(join(outdir, "random_comparison_R_%.1f.png"%cdf_ratio), bbox_inches="tight")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", dest="training_conf", help="Path to training conf dir", required=True)
    parser.add_argument("-U", dest="frame_utils", help="Path to frame_utils.csv calculated using utility based method", required=True)
    parser.add_argument("-O", dest="outdir", help="Path to output directory", default="/tmp")
    parser.add_argument("-R", dest="cdf_ratio", help="Ratio of video frames to build CDF. Metrics on the rest", default=0.5, type=float)
    
    args = parser.parse_args()

    main(args.training_conf, args.outdir, args.frame_utils, args.cdf_ratio)
