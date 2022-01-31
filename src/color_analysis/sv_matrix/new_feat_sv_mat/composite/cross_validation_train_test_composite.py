import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../../../../"))
sys.path.insert(0, os.path.join(script_dir, "../../../"))
sys.path.insert(0, os.path.join(script_dir, "../../"))
sys.path.insert(0, os.path.join(script_dir, "../"))
sys.path.insert(0, os.path.join(script_dir, "../../composite"))
import python_server.mapping_features
import object_based_metrics_calc
import yaml
from os.path import isfile, join, isdir
from os import listdir
import pandas as pd
import aggregate_sv_matrix
import compute_test_frames_util
import compute_composite_utils
import new_feat_sv_mat.cross_validation_train_test as cv_single_color
import compute_composite_utils
import extract_sv_matrix_from_bin

def main(training_conf_dirs, mats_dir, composite_or):
    training_dirs = []
    colors = []
    # Read the conf
    confs = []
    for training_conf_dir in training_conf_dirs:
        with open(join(training_conf_dir, "conf.yaml")) as fi:
            conf = yaml.safe_load(fi)
        training_dir = conf["training_dir"]
        num_bins = conf["num_bins"]
        confs.append(conf)
        training_dirs.append(training_dir)

        # Extract color from the conf
        for hue_bin in conf["hue_bins"]:
            color = hue_bin["name"]
            break
        colors.append(color)

    vids = compute_composite_utils.get_common_videos(training_dirs[0], training_dirs[1])
    cv_folds = 2
    num_test_vids = int(len(vids)/cv_folds)
    merged_dfs = []
    for cv_fold in range(cv_folds):
        low = cv_fold*num_test_vids
        high = (cv_fold+1)*num_test_vids
        if cv_fold == cv_folds-1: # Last fold
            if high < len(vids):
                high = len(vids)
        test_vids = [vids[idx] for idx in range(len(vids)) if idx >= low and idx < high]
        train_vids = [vids[idx] for idx in range(len(vids)) if idx < low or idx >= high]

        print (test_vids, train_vids)

        color_dfs = []
        for color_idx in range(len(colors)):
            conf = confs[color_idx]
            conf_colors = extract_sv_matrix_from_bin.get_conf_colors(conf)
            raw_data = []
            color = colors[color_idx]
            training_dir = training_dirs[color_idx]
            vid_mat_dir = join(mats_dir, color)
            ## Seperate out the positive and negative matrices
            positive_mats = []
            negative_mats = []
            for train_vid in train_vids:
                vid_dir = join(vid_mat_dir, train_vid)
                positive_mats.append(join(vid_dir, "sv_matrix_label_True_BINS_%d_C_%s.txt"%(num_bins, color)))
                negative_mats.append(join(vid_dir, "sv_matrix_label_False_BINS_%d_C_%s.txt"%(num_bins, color)))
    
            aggr_mat, max_val = aggregate_sv_matrix.aggregate(num_bins, positive_mats, negative_mats)
   
            # Compute normalization factor over training data
            max_util = None
            for train_vid in train_vids:
                train_vid_dir = join(training_dir, train_vid)
                bin_file = [join(train_vid_dir, f) for f in listdir(train_vid_dir) if isfile(join(train_vid_dir, f)) and f.endswith(".bin")][0]
                mats = cv_single_color.read_per_frame_mats(bin_file, conf_colors, conf["num_bins"], conf["pf_threshold"])
                for mat in mats:
                    util  = compute_test_frames_util.compute_util(mat, aggr_mat[True])
                    if max_util == None or max_util < util:
                        max_util = util
  
            # Now compute the utility for test frames
            for test_vid in test_vids:
                test_vid_dir = join(training_dir, test_vid)
                bin_file = [join(test_vid_dir, f) for f in listdir(test_vid_dir) if isfile(join(test_vid_dir, f)) and f.endswith(".bin")][0]
                ground_truth_frames = python_server.mapping_features.read_samples(bin_file)
                mats = cv_single_color.read_per_frame_mats(bin_file, conf_colors, conf["num_bins"], conf["pf_threshold"])
                frame_idx = 0
                for mat in mats:
                    label = ground_truth_frames[frame_idx].label
                    count = ground_truth_frames[frame_idx].detections.totalDetections
                    # Already normalized util
                    util  = compute_test_frames_util.compute_util(mat, aggr_mat[True])/max_util # TODO Should we normalize with max util of the given video?
                    raw_data.append([count, label, frame_idx, util, test_vid, color])
                    frame_idx += 1
    
            df = pd.DataFrame(raw_data, columns=["count_%d"%color_idx, "label_%d"%color_idx, "frame_id", "utility_%d"%color_idx, "vid_name", "color_%d"%color_idx])
            df["vid_frame"] = df.apply(lambda row: row.vid_name+"_"+str(row.frame_id), axis=1)
            color_dfs.append(df)

        merged = pd.merge(color_dfs[0], color_dfs[1], on="vid_frame")
        merged["cv_fold"] = cv_fold

        if composite_or:
            merged["composite_utility"] = merged.apply(lambda row: max(row.utility_0, row.utility_1), axis=1)
            merged["composite_label"] = merged.apply(lambda row: row.label_0 or row.label_1, axis=1)
        else:
            merged["composite_utility"] = merged.apply(lambda row: min(row.utility_0, row.utility_1), axis=1)
            merged["composite_label"] = merged.apply(lambda row: row.label_0 and row.label_1, axis=1)

        merged_dfs.append(merged)

    df = pd.concat(merged_dfs, ignore_index=True)
    df.to_csv(join(mats_dir, "frame_utils.csv"))

    plt.close()
    fig, ax = plt.subplots(figsize=(24,8))
    sns.boxplot(data=df, x="vid_name_x", y="composite_utility", hue="composite_label", ax=ax)
    ax.set_xlabel("Label of the frame")
    ax.set_ylabel("Utility for colors %s"%colors)
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
    fig.savefig(join(mats_dir, "composite_util_CROSSVIDEO.png"), bbox_inches="tight")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", dest="training_confs", help="Directories containing training conf", nargs="+")
    parser.add_argument("-M", dest="mats_dir", help="Directory containing per frame matrix")
    parser.add_argument("--composite-or", help="Boolean flag of whether the composition is OR. Otherwise its assumed as AND.", dest="composite_or", action="store_true")

    args = parser.parse_args()

    main(args.training_confs, args.mats_dir, args.composite_or)
