import argparse
from os.path import join, isfile, isdir
from os import listdir
import os, sys
import math
import pandas as pd
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../"))
import extract_sv_matrix

def read_util_mat_file(mat_file, norm_factor):
    mat = []
    with open(mat_file) as f:
        for line in f.readlines():
            mat.append([])
            s = line.split()
            for x in s:
                mat[-1].append(float(x)/norm_factor)
    return mat

def main(training_confs, video_dirs, composite_or, outdir, training_split):
    if len(training_confs) != len(video_dirs):
        print ("Length of util_mats does not equal length of video_dirs")
        exit(0)

    util_mats_files = []
    norm_factors = []
    for training_conf_dir in training_confs:
        util_mats_files.append([join(training_conf_dir, x) for x in listdir(training_conf_dir) if isfile(join(training_conf_dir, x)) and "utils_" in x and ".txt" in x][0])

        with open(join(training_conf_dir, "normalization_factor")) as f:
            norm_factors.append(float(f.readline()))

    util_mats = []
    for color_idx in range(len(util_mats_files)):
        util_mats.append(read_util_mat_file(util_mats_files[color_idx], norm_factors[color_idx]))

    # Now read the utility values of all frames
    aggr_utils = []
    color_idx = 0
    for video_dir in video_dirs:
        df = pd.read_csv(join(video_dir, "frame_utils.csv"))
        norm = norm_factors[color_idx]

        for idx, row in df.iterrows():
            u = row["utility"]/norm
            if len(aggr_utils) <= idx:
                aggr_utils.append(u)
            else:
                if composite_or:
                    aggr_utils[idx] = max(aggr_utils[idx], u)
                else: # AND
                    aggr_utils[idx] = min(aggr_utils[idx], u)

        color_idx += 1

    num_training_frames = int(len(aggr_utils)*training_split)

    util_vals = sorted(aggr_utils[:num_training_frames])
    cdf_points = 100
    idx_steps = math.ceil(len(util_vals)/cdf_points)

    curr_idx = 0
    with open(join(outdir, "util_cdf.txt"), "w") as f:
        while curr_idx < len(util_vals):
            f.write("%f\t%f\n"%(curr_idx/len(util_vals), util_vals[curr_idx]))
            curr_idx += idx_steps
        f.write("%f\t%f\n"%(1.0, util_vals[-1]))

    for color_idx in range(len(util_mats)):
        extract_sv_matrix.dump_sv_mat(util_mats[color_idx], join(outdir, "util_mat_color_%d.txt"%color_idx))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", help="Directories of training configurations", nargs="+", dest="training_confs", required=True)
    parser.add_argument("-V", help="Video dirs", nargs="+", dest="video_dirs", required=True)
    parser.add_argument("-O", help="Output directory to store the model", dest="outdir", required=True)
    parser.add_argument("-T", help="Training split fraction", dest="training_split", required=True, type=float)
    parser.add_argument("--composite-or", help="Boolean flag of whether the composition is OR. Otherwise its assumed as AND.", dest="composite_or", action="store_true")
    args = parser.parse_args()
    

    main(args.training_confs, args.video_dirs, args.composite_or, args.outdir, args.training_split)
