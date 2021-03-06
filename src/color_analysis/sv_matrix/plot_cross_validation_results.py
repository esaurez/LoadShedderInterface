import argparse
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from os.path import join, isfile
from matplotlib.font_manager import FontProperties
import sys, os
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../"))
import object_based_metrics_calc

def plot_utils(df, outdir, fontsize, label_col="label", utility_col="utility", vid_name_col="vid_name"):
    plt.close()
    fig, ax = plt.subplots(figsize=(16,6))
    df.loc[df[label_col] == False, label_col] = "-ve"
    df.loc[df[label_col] == True, label_col] = "+ve"
    sns.boxplot(data=df, x=vid_name_col, y=utility_col, hue=label_col, ax=ax)
    num_vids = len(df[vid_name_col].unique())
    fontP = FontProperties()
    fontP.set_size(fontsize)
    ax.set_xlabel("Video file", fontsize=fontsize)
    ax.set_ylabel("Utility", fontsize=fontsize)
    for idx in range(len(ax.get_xticklabels())):
        print (idx+1, ax.get_xticklabels()[idx].get_text())
    ax.set_xticklabels(range(1, num_vids+1))
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    legend = ax.legend(fancybox=True,shadow=False,title='Frame label', prop=fontP, ncol=2)
    plt.setp(legend.get_title(),fontsize=fontsize)
    fig.savefig(join(outdir, "combined_util_CROSSVIDEO.png"), bbox_inches="tight")

def plot_rates(df, outdir, fontsize, obj_files, label_col="label", utility_col="utility", vid_name_col="vid_name"):
    # Build a map for (vid, cv_fold) --> [utils] for fast access
    utils_map = {}
    grouping = df.groupby([vid_name_col, "cv_fold"])
    for group in grouping.groups.keys():
        (vid, fold) = group
        gdf = grouping.get_group(group)
        utils_map[group] = [(row[utility_col], row[label_col]) for idx, row in gdf.iterrows()]

    num_points = 50
    max_util = df[utility_col].max() 
    frame_drop_rates = []
    obj_det_rates = []
    false_neg_rates = []
    obj_sel_rates = []
    util_thresholds = np.arange(0, max_util, max_util/num_points)
    for util_threshold in util_thresholds:
        per_threshold_obj_sel_rates = []
        total_frames = 0
        frames_dropped = 0
        total_objs = 0
        detected_objs = 0
        false_negs = 0
        ground_truth_positives = 0
        grouping = df.groupby([vid_name_col, "cv_fold"])
        for group in grouping.groups.keys():
            (vid, fold) = group
            gdf = grouping.get_group(group)
            obj_frames = obj_files[vid]
            is_frame_dropped = [u < util_threshold for (u,l) in utils_map[group]]
            obj_covered = object_based_metrics_calc.get_obj_coverage(obj_frames, is_frame_dropped, 0)
            obj_frame_sel_rates = object_based_metrics_calc.get_obj_frame_sel_rates(obj_frames, is_frame_dropped, 0)
            per_threshold_obj_sel_rates += obj_frame_sel_rates
            total_frames += len(is_frame_dropped)
            frames_dropped += len([x for x in is_frame_dropped if x])
            total_objs += len(obj_covered)
            detected_objs += len([x for x in obj_covered if obj_covered[x]])

            # number of false negatives
            false_negs += len([idx for idx in range(len(is_frame_dropped)) if is_frame_dropped[idx] and utils_map[group][idx][1]=="+ve"])
            # number of ground truth +ve frames
            ground_truth_positives += len([l for (u,l) in utils_map[group] if l=="+ve"])

        obj_det_rates.append(detected_objs/float(total_objs))
        frame_drop_rates.append(frames_dropped/float(total_frames))
        false_neg_rates.append(false_negs/float(ground_truth_positives))
        obj_sel_rates.append(np.mean(per_threshold_obj_sel_rates))

    fontP = FontProperties()
    fontP.set_size(fontsize)
    plt.close()
    fig, ax = plt.subplots(figsize=(8,6))
    #ax.plot(util_thresholds, false_neg_rates, label="False -ve rate")
    ax.plot(util_thresholds, frame_drop_rates, label="Frame Drop Rate")
    #ax.plot(util_thresholds, obj_det_rates, label="Obj Detn. Rate")
    ax.plot(util_thresholds, obj_sel_rates, label="Object-based QoR")
    ax.set_xlabel("Utility threshold", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    legend = ax.legend(fancybox=True,shadow=False, prop=fontP)
    fig.savefig(join(outdir, "rates.png"), bbox_inches="tight")

def main(frame_utils, outdir, training_conf_dir):
    df = pd.read_csv(frame_utils)
    fontsize = 20

    # Now plotting the object based metric
    with open(join(training_conf_dir, "conf.yaml")) as fi:
        conf = yaml.safe_load(fi)

    plot_utils(df, outdir, fontsize)

    training_dir = conf["training_dir"]
    vids = df["vid_name"].unique()
    obj_files = {}
    for vid in vids:
        if vid not in obj_files:
            obj_files[vid] = {}
        obj_file = join(join(training_dir, vid), "unique_objs_per_frame.txt")
        vid_objs = object_based_metrics_calc.get_obj_frames(obj_file)
        for obj in vid_objs:
            vid_obj_id = str(obj)
            obj_files[vid][vid_obj_id] = vid_objs[obj]

    plot_rates(df, outdir, fontsize, obj_files) 


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-U", dest="frame_utils", help="Path to frame_utils.csv generated by cross_validation")
    parser.add_argument("-O", dest="outdir", help="Output directory")
    parser.add_argument("-C", dest="training_conf", help="Directory containing training conf")

    args = parser.parse_args()

    main(args.frame_utils, args.outdir, args.training_conf)

