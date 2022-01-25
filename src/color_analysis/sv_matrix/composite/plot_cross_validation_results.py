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
sys.path.insert(0, os.path.join(script_dir, "../../"))
import object_based_metrics_calc

def main(frame_utils, outdir, composite_or, training_confs):
    df = pd.read_csv(frame_utils)
    plt.close()
    fig, ax = plt.subplots(figsize=(16,6))
    df.loc[df.composite_label == False, "composite_label"] = "-ve"
    df.loc[df.composite_label == True, "composite_label"] = "+ve"
    sns.boxplot(data=df, x="vid_name_x", y="composite_utility", hue="composite_label", ax=ax)
    num_vids = len(df["vid_name_x"].unique())
    fontsize = 20
    fontP = FontProperties()
    fontP.set_size(fontsize)
    ax.set_xlabel("Video file", fontsize=fontsize)
    ax.set_ylabel("Utility", fontsize=fontsize)
    ax.set_xticklabels(range(1, num_vids+1))
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    legend = ax.legend(fancybox=True,shadow=False,title='Frame label', prop=fontP, ncol=2)
    plt.setp(legend.get_title(),fontsize=fontsize)
    fig.savefig(join(outdir, "composite_cross_val__util_CROSSVIDEO.png"), bbox_inches="tight")

    if not composite_or:
        print ("Object based metrics only possible with composite_or so far")
        return

    vids = df["vid_name_x"].unique()
    obj_files = {}
    color = 0
    for training_conf_dir in training_confs:
        with open(join(training_conf_dir, "conf.yaml")) as fi:
            conf = yaml.safe_load(fi)
        training_dir = conf["training_dir"]
        for vid in vids:
            if vid not in obj_files:
                obj_files[vid] = {}
            obj_file = join(join(training_dir, vid), "unique_objs_per_frame.txt")
            vid_objs = object_based_metrics_calc.get_obj_frames(obj_file)
            for obj in vid_objs:
                vid_obj_id = str(color)+"_"+str(obj)
                obj_files[vid][vid_obj_id] = vid_objs[obj]
        color+=1

    # Build a map for (vid, cv_fold) --> [utils] for fast access
    utils_map = {}
    grouping = df.groupby(["vid_name_x", "cv_fold"])
    for group in grouping.groups.keys():
        (vid, fold) = group
        gdf = grouping.get_group(group)
        utils_map[group] = [(row["composite_utility"], row["composite_label"]) for idx, row in gdf.iterrows()]

    num_points = 50
    max_util = df["composite_utility"].max()
    frame_drop_rates = []
    obj_det_rates = []
    false_neg_rates = []
    util_thresholds = np.arange(0, max_util, max_util/num_points)
    for util_threshold in util_thresholds:
        total_frames = 0
        frames_dropped = 0
        total_objs = 0
        detected_objs = 0
        false_negs = 0
        ground_truth_positives = 0
        grouping = df.groupby(["vid_name_x", "cv_fold"])
        for group in grouping.groups.keys():
            (vid, fold) = group
            gdf = grouping.get_group(group)
            obj_frames = obj_files[vid]

            is_frame_dropped = [u < util_threshold for (u,l) in utils_map[group]]
            obj_covered = object_based_metrics_calc.get_obj_coverage(obj_frames, is_frame_dropped, 0)
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

    plt.close()
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(util_thresholds, obj_det_rates, color="red")
    ax.set_xlabel("Utility threshold", fontsize=fontsize)
    ax.set_ylabel("Fraction of objects detected", fontsize=fontsize)
    ax2 = ax.twinx()
    ax2.plot(util_thresholds, frame_drop_rates, color="blue")
    ax2.set_ylabel("Fraction of frames dropped", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=fontsize)
    fig.savefig(join(outdir, "object_based_drops.png"), bbox_inches="tight")

    # Frame based results
    plt.close()
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(util_thresholds, false_neg_rates, color="red")
    ax.set_xlabel("Utility threshold", fontsize=fontsize)
    ax.set_ylabel("False-negative rate", fontsize=fontsize)
    ax2 = ax.twinx()
    ax2.plot(util_thresholds, frame_drop_rates, color="blue")
    ax2.set_ylabel("Fraction of frames dropped", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=fontsize)
    fig.savefig(join(outdir, "false-negative-rates.png"), bbox_inches="tight")

    plt.close()
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(util_thresholds, false_neg_rates, label="False -ve rate")
    ax.plot(util_thresholds, frame_drop_rates, label="Frame Drop Rate")
    ax.plot(util_thresholds, obj_det_rates, label="Obj Detn. Rate")
    ax.set_xlabel("Utility threshold", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    legend = ax.legend(fancybox=True,shadow=False, prop=fontP)
    fig.savefig(join(outdir, "rates.png"), bbox_inches="tight")
 
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-U", dest="frame_utils", help="Path to frame_utils.csv generated by cross_validation")
    parser.add_argument("-O", dest="outdir", help="Output directory")
    parser.add_argument("--composite-or", help="Boolean flag of whether the composition is OR. Otherwise its assumed as AND.", dest="composite_or", action="store_true")
    parser.add_argument("-C", dest="training_confs", help="Directories containing training conf", nargs="+")

    args = parser.parse_args()

    main(args.frame_utils, args.outdir, args.composite_or, args.training_confs)
