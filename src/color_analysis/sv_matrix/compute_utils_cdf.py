import argparse
from os import listdir
from os.path import join, isfile, isdir
import yaml
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import math

def aggr_and_write_utils(aggr_df, outdir):
    max_utils = []

    util_vals = []
    grouping = aggr_df.groupby(["frame_id", "vid_name"])
    for key in grouping.groups.keys():
        (frame_id, vid_name) = key
        gdf = grouping.get_group(key)

        aggr = gdf.agg({"utility":"max","label":"max"})

        max_utils.append([frame_id, vid_name, aggr["utility"], aggr["label"]])

        util_vals.append(aggr["utility"])

    max_utils_df = pd.DataFrame(max_utils, columns=["frame_id", "vid_name", "utility", "label"])

    plt.close()
    sns.ecdfplot(data=max_utils_df, x="utility")
    plt.savefig(join(outdir, "utils_cdf.png"), bbox_inches="tight")

    util_vals = sorted(util_vals)
    cdf_points = 100
    idx_steps = math.ceil(len(util_vals)/cdf_points)

    curr_idx = 0
    with open(join(outdir, "util_cdf.txt"), "w") as f:
        while curr_idx < len(util_vals):
            f.write("%f\t%f\n"%(curr_idx/len(util_vals), util_vals[curr_idx]))
            curr_idx += idx_steps
        f.write("%f\t%f\n"%(1.0, util_vals[-1]))

def main(training_conf, outdir):
    with open(training_conf) as f:
        config = yaml.safe_load(f)
    video_dir = config["training_dir"]
    training_split = config["training_split"]
    
    # Now get all the frame utility files
    video_dirs = [join(video_dir, o) for o in listdir(video_dir) if isdir(join(video_dir, o))]
    dfs = []
    for vid_dir in video_dirs:
        try:
            df = pd.read_csv(join(vid_dir, "frame_utils.csv"))
            num_train_frames = int(len(df)*training_split)
            df = df[df["frame_id"]<num_train_frames]
            aggr_and_write_utils(df, vid_dir)
            dfs.append(df)
        except:
            pass

    aggr_df = pd.concat(dfs, ignore_index=True)
    aggr_and_write_utils(aggr_df, outdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", dest="training_conf", help="Training configuration file")
    parser.add_argument("-O", dest="outdir", help="Output directory")

    args = parser.parse_args()
    main(args.training_conf, args.outdir)
