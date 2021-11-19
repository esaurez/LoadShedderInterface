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

def main(training_conf, outdir):
    with open(training_conf) as f:
        config = yaml.safe_load(f)
    video_dir = config["training_dir"]
    
    # Now get all the frame utility files
    video_dirs = [join(video_dir, o) for o in listdir(video_dir) if isdir(join(video_dir, o))]
    util_csvs = []
    for vid_dir in video_dirs:
        util_csvs += [join(vid_dir, f) for f in listdir(vid_dir) if isfile(join(vid_dir, f)) and f=="frame_utils.csv"]

    dfs = []
    for util_csv in util_csvs:
        df = pd.read_csv(util_csv)
        dfs.append(df)
    aggr_df = pd.concat(dfs, ignore_index=True)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", dest="training_conf", help="Training configuration file")
    parser.add_argument("-O", dest="outdir", help="Output directory")

    args = parser.parse_args()
    main(args.training_conf, args.outdir)
