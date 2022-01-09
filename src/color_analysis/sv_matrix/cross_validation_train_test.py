import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../../"))
import python_server.mapping_features
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
            bin_file = [join(test_vid_dir, f) for f in listdir(test_vid_dir) if isfile(join(test_vid_dir, f)) and f.endswith(".bin")][0]
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

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", dest="training_conf", help="Directory containing training conf")
    parser.add_argument("-M", dest="mats_dir", help="Directory containing per frame matrix")
    args = parser.parse_args()

    main(args.training_conf, args.mats_dir)
