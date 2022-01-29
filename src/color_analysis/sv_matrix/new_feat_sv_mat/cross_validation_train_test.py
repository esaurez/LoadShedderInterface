import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../../../"))
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
import extract_sv_matrix_from_bin

def read_per_frame_mats(bin_file, conf_colors, num_bins, pf_threshold):
    ground_truth_frames = python_server.mapping_features.read_samples(bin_file)
    mats = []
    # Creating the ProcessPoolExecutor
    executor = ProcessPoolExecutor(max_workers=32)
    num_total_frames = len(ground_truth_frames)

    futures = []
    for frame_idx in range(num_total_frames):
        frame = ground_truth_frames[frame_idx]
        future = executor.submit(extract_sv_matrix_from_bin.get_sv_counts, frame, conf_colors, num_bins, pf_threshold)
        futures.append(future)

    for future in futures:
        mats.append(future.result()[0])
    return mats

def main(training_conf_dir, mats_dir):
    # Read the conf
    with open(join(training_conf_dir, "conf.yaml")) as fi:
        conf = yaml.safe_load(fi)
    training_dir = conf["training_dir"]
    num_bins = conf["num_bins"]
    pf_threshold = conf["pf_threshold"]
    conf_colors = extract_sv_matrix_from_bin.get_conf_colors(conf)
    if len(conf_colors) > 1:
        print ("Not more than 1 color supported. Exiting.")
        exit(1)
    if "is_composite" in conf and conf["is_composite"] == "true":
        print ("Composite query not supported. Exiting.")
        exit(1)

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
        util_mat = []
        for row in range(len(aggr_mat[True])):
            util_mat.append([])
            for col in range(len(aggr_mat[True][row])):
                util_mat[-1].append(aggr_mat[True][row][col])
                #util_mat[-1].append(aggr_mat[True][row][col] - aggr_mat[False][row][col])

        # Now compute the utility for test frames
        for test_vid in test_vids:
            test_vid_dir = join(training_dir, test_vid)
            bin_file = [join(test_vid_dir
, f) for f in listdir(test_vid_dir) if isfile(join(test_vid_dir, f)) and f.endswith(".bin")][0]
            ground_truth_frames = python_server.mapping_features.read_samples(bin_file)
            mats = read_per_frame_mats(bin_file, conf_colors, num_bins, pf_threshold)
            frame_idx = 0
            for mat in mats:
                label = ground_truth_frames[frame_idx].label
                count = ground_truth_frames[frame_idx].detections.totalDetections
                util  = compute_test_frames_util.compute_util(mat, util_mat)
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

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", dest="training_conf", help="Directory containing training conf")
    parser.add_argument("-M", dest="mats_dir", help="Directory containing per frame matrix")
    args = parser.parse_args()

    main(args.training_conf, args.mats_dir)
