import argparse
import os, sys
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../"))
import model.build_model
import python_server.mapping_features
from capnp_serial import mapping_capnp
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from os.path import basename, join
import math

# TODO : Fix this definition of colors everywhere. Centralize it in a constants.py file.
# The main distribution of color values is in mapping_features file
COLORS = ['red', 'orange','yellow','spring_green','green','ocean_green','light_blue','blue','purple','magenta']

def main(training_data, outdir):
    raw_values = [] # array to contain the raw values which will later be converted to dataframe
    video_files = glob.glob(training_data+"/*.bin")
    for video_file in video_files:
        frame_id = 0
        # Extracting the features from training data samples
        training_samples = python_server.mapping_features.read_samples(video_file)

        for sample in training_samples:
            if frame_id < 25:
                frame_id += 1
                continue

            label = sample.label
            feat = sample.feats.feats[0]
            if feat.type == mapping_capnp.Feature.Type.fullColorHistogram:
                histogram = feat.feat.wholeHisto.histo

                total_pixels = histogram.totalCountedPixels
                raw_values.append([frame_id, total_pixels, basename(video_file), label])

            frame_id += 1

    # DataFrame to contain the aggregate data
    df = pd.DataFrame(raw_values, columns=["frame_id", "total_pixels", "video_file", "label"])

    plt.close()
    sns.lineplot(data=df, x="frame_id", y="total_pixels", hue="video_file")
    plt.legend()
    plt.savefig(join(outdir, "total_pixels.png"), bbox_inches="tight")

    vid_grouping = df.groupby("video_file")
    num_groups = len(vid_grouping.groups.keys())
    cols = 2
    rows = math.ceil(num_groups/cols)
    fig, axs = plt.subplots(rows, cols, figsize=(10*cols, 2*rows))
    plot_idx = 0
    for vid in vid_grouping.groups.keys():
        row = int(plot_idx/cols)
        col = plot_idx - cols*row
        vid_df = vid_grouping.get_group(vid)
        plt.close()
        if rows == 1:
            ax = axs[col]
        sns.scatterplot(data=vid_df, x="frame_id", y="total_pixels", style="video_file", hue="label", size=1, ax=ax)
        ax.get_legend().remove()
        ax.text(.5,.9, vid, horizontalalignment='center', transform=ax.transAxes)
        plot_idx +=1 
    fig.savefig(join(outdir, "total_pixels_labeled.png"), bbox_inches="tight")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzing the distribution of Pixel Fraction")
    parser.add_argument("-D", "--training-data", dest="training_data", help="Path to training data")
    parser.add_argument("-O", "--out-dir", dest="output_dir", help="Path to output data that would contain plots")
    
    args = parser.parse_args()
    main(args.training_data, args.output_dir)
