import argparse
import os, sys
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../../"))
import model.build_model
import python_server.mapping_features
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
import glob
from os import listdir
from os.path import basename, join, isdir, isfile
import math

# TODO : Fix this definition of colors everywhere. Centralize it in a constants.py file.
# The main distribution of color values is in mapping_features file
COLORS = ['red', 'orange','yellow','spring_green','green','ocean_green','light_blue','blue','purple','magenta']

COLORS_TO_FILTER = ["red"]
#COLORS_TO_FILTER = ["red", "magenta", "blue", "green"]

def main(training_data, outdir):
    video_files = []
    for vid_dir in listdir(training_data):
        if not isdir(join(training_data, vid_dir)):
            continue
        D = join(training_data, vid_dir)
        for f in listdir(D):
            if isfile(join(D, f)) and f.endswith(".bin"):
                video_files.append(join(D, f))
    #video_files = glob.glob(training_data+"/*.bin")
    raw_values = [] # array to contain the raw values which will later be converted to dataframe

    video_file_idx = 0
    for video_file in video_files:
        video_file_idx += 1
        for absolute_pixel_count in [False]:#[True, False]:

            # Extracting the features from training data samples
            observations = python_server.mapping_features.read_training_file(video_file, absolute_pixel_count=absolute_pixel_count)
            frame_id=0
            for observation in observations:
                if frame_id <= 5:
                    frame_id += 1
                    continue
                # observation is in the form of [feat1, feat2, ..., featN, label]
                feature_list = observation[0:-1]
                label = observation[-1]

                color_count = 0
                for color in COLORS:
                    color_position = COLORS.index(color)
                    pixel_fraction = feature_list[color_position]

                    if color in COLORS_TO_FILTER:
                        raw_values.append([color, pixel_fraction, label, frame_id, video_file_idx, absolute_pixel_count])
                    color_count += 1
                
                frame_id += 1
        
            # DataFrame to contain the aggregate data
    df = pd.DataFrame(raw_values, columns=["color", "pixel_fraction", "label", "frame_id", "video_file", "absolute_pixel_count"])

    '''
    abs_grouping = df.groupby(["absolute_pixel_count", "video_file"])
    for key in abs_grouping.groups.keys():
        abs_gdf = abs_grouping.get_group(key)
        (abs_pixel_count, video_file) = key
        grouping = abs_gdf.groupby(["color"])
        num_plots = len(grouping.groups.keys())
        num_cols = 2
        num_rows = math.ceil(num_plots/num_cols)
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(12*num_cols,4*num_rows))
        idx = 0
        for key in grouping.groups.keys():
            row = int (idx / num_cols)
            col = idx - row*num_cols
            ax = axs[row][col]
            gdf = grouping.get_group(key)
            plt.close()
            sns.scatterplot(data=gdf, x="frame_id", y="pixel_fraction", hue="label", size=1, ax=ax)
            ax.text(.5,.9, key, horizontalalignment='center', transform=ax.transAxes)
            ax.get_legend().remove()
            if abs_pixel_count:
                ax.set_ylabel("Absolute Pixel Count")
            else:
                ax.set_ylabel("Pixel Fraction")
            ax.set_xlabel("Frame ID (1 frame per second)")
            idx += 1
        fig.savefig(join(outdir, "pixel_fraction_vs_%s_abs_count_%s.png"%(video_file, str(abs_pixel_count))), bbox_inches="tight")
    '''
    print ("Now plotting")
    upper_grouping = df.groupby(["absolute_pixel_count"])
    for upper_group_key in upper_grouping.groups.keys():
        (absolute_pixel_count) = upper_group_key
        upper_group_df = upper_grouping.get_group(upper_group_key)
        grouping = upper_group_df.groupby(["color"])

        num_keys = len(grouping.groups.keys())
        cols = 2
        rows = math.ceil(num_keys/float(cols))
        plt.close()
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)
        fig, axs = plt.subplots(rows, cols, figsize=(9*cols,3*rows))

        row = 0
        col = 0
        for key in grouping.groups.keys():
            if rows == 1:
                ax = axs[col]
            else:
                ax = axs[row][col]
            (color) = key
            gdf = grouping.get_group(key)
            gdf.loc[gdf.label == 0, "label"] = "-ve"
            gdf.loc[gdf.label == 1, "label"] = "+ve"
            plt.close()
            sns.stripplot(data=gdf, x="video_file", y="pixel_fraction", hue="label", dodge=True, ax=ax)
            ax.text(.5,.9, key, horizontalalignment='center', transform=ax.transAxes)
            fig1, ax1 = plt.subplots(figsize=(9,3))
            sns.stripplot(data=gdf, x="video_file", y="pixel_fraction", hue="label", dodge=True, ax=ax1)
            fontsize = 16
            fontP = FontProperties()
            fontP.set_size(fontsize)
            for axis in [ax, ax1]:
                axis.set_xlabel("Video file", fontsize=fontsize)
                if absolute_pixel_count:
                    axis.set_ylabel("Absolute count of pixels", fontsize=fontsize)
                else:
                    axis.set_ylabel("Hue Fraction for %s"%(color.title()), fontsize=fontsize)
                    #axis.set_ylabel("Hue Fraction for %s\n(num %s pixels / num foreground pixels)"%(color.title(), color), fontsize=fontsize)
                axis.tick_params(axis='both', which='major', labelsize=fontsize)

                legend = ax1.legend(fancybox=True,shadow=False,title='Frame label', prop=fontP, ncol=2)
                plt.setp(legend.get_title(),fontsize=fontsize)

            #ax.title.set_text("Abs pixel count = %s; Video File = %s"%(str(absolute_pixel_count), video_file))
            fig1.savefig(join(outdir, "huefrac_%s_scatter_abs_count_%s.png"%(color, str(absolute_pixel_count))), bbox_inches="tight")

            col += 1
            if col >= cols:
                col = 0
                row += 1

        fig.savefig(join(outdir, "pixel_fraction_scatter_abs_count_%s.png"%(str(absolute_pixel_count))), bbox_inches="tight")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzing the distribution of Pixel Fraction")
    parser.add_argument("-D", "--training-data", dest="training_data", help="Path to training data")
    parser.add_argument("-O", "--out-dir", dest="output_dir", help="Path to output data that would contain plots")
    
    args = parser.parse_args()
    main(args.training_data, args.output_dir)
