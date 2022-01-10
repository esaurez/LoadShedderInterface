import argparse
import yaml
from os import listdir
from os.path import join, isfile, isdir
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def get_common_videos(dir1, dir2):
    vids1 = [x for x in listdir(dir1) if isdir(join(dir1, x))]
    vids2 = [x for x in listdir(dir2) if isdir(join(dir2, x))]

    m = {}
    for v in vids1:
        m[v] = False
    for v in vids2:
        if v in m:
            m[v] = True

    common = [v for v in m if m[v]]
    return common

def main(training_confs, outdir, composite_or):
    if len(training_confs) != 2:
        print ("Number of colors has to be 2")
        exit(1)

    training_dirs = []
    norm_factors = []
    color_names = []
    training_split = None
    for training_conf_dir in training_confs:
        with open(join(training_conf_dir, "conf.yaml")) as f:
            conf = yaml.safe_load(f)
            training_dirs.append(conf["training_dir"])
            training_split = conf["training_split"]
            color_names.append(conf["color_names"])
        with open(join(training_conf_dir, "normalization_factor")) as f:
            norm_factors.append(float(f.readline()))

    common_vids = get_common_videos(training_dirs[0], training_dirs[1])
    
    # Now go through each video one by one
    cols = ["color", "count", "label", "utility"] # Cols that are video-specific
    merged_dfs =[]
    for vid in common_vids:
        dfs = []
        for color in [0,1]:
            vid_dir = join(training_dirs[color], vid)
            df = pd.read_csv(join(vid_dir, "frame_utils.csv"))
            # Normalizing the utility value
            df['utility'] = df['utility'].apply(lambda x: x/norm_factors[color])
            col_rename_map = {}
            for col in cols:
                col_rename_map[col] = "%s_%d"%(col, color)
            df.rename(columns=col_rename_map, inplace=True)
            dfs.append(df)
        merged = pd.merge(dfs[0], dfs[1], on="frame_id")
        if composite_or:
            merged["composite_utility"] = merged.apply(lambda row: max(row.utility_0, row.utility_1), axis=1)
            merged["composite_label"] = merged.apply(lambda row: row.label_0 or row.label_1, axis=1)
        else:
            merged["composite_utility"] = merged.apply(lambda row: min(row.utility_0, row.utility_1), axis=1)
            merged["composite_label"] = merged.apply(lambda row: row.label_0 and row.label_1, axis=1)
    
        # Remove training frames
        num_training_frames = int(training_split*len(merged))
        merged = merged[merged["frame_id"] >= num_training_frames]
        merged_dfs.append(merged)

    df = pd.concat(merged_dfs, ignore_index=True)
    
    fig, ax = plt.subplots(figsize=(24,8))
    sns.boxplot(data=df, x="vid_name_x", y="composite_utility", hue="composite_label", ax=ax)
    ax.set_xlabel("Video")
    ax.set_ylabel("Composite utility for %s"%color_names)
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
    fig.savefig(join(outdir, "combined_util_CROSSVIDEO.png"), bbox_inches="tight")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", help="Directories of training configurations", nargs="+", dest="training_confs", required=True)
    parser.add_argument("-O", help="Output directory to store the utility", dest="outdir", required=True)
    parser.add_argument("--composite-or", help="Boolean flag of whether the composition is OR. Otherwise its assumed as AND.", dest="composite_or", action="store_true")

    args = parser.parse_args()

    main(args.training_confs, args.outdir, args.composite_or)
