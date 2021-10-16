import argparse
import os, sys
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../"))
import model.build_model
import python_server.mapping_features
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from os.path import basename, join

# TODO : Fix this definition of colors everywhere. Centralize it in a constants.py file.
# The main distribution of color values is in mapping_features file
COLORS = ['red', 'orange','yellow','spring_green','green','ocean_green','light_blue','blue','purple','magenta']

def main(training_data, outdir):
    video_files = glob.glob(training_data+"/*.bin")
    for video_file in video_files:
        for absolute_pixel_count in [True, False]:
            raw_values = [] # array to contain the raw values which will later be converted to dataframe
            # Extracting the features from training data samples
            observations = python_server.mapping_features.read_training_file(video_file, absolute_pixel_count=absolute_pixel_count)
            frame_id=0
            for observation in observations:
                if frame_id < 25:
                    frame_id += 1
                    continue
                # observation is in the form of [feat1, feat2, ..., featN, label]

                feature_list = observation[0:-1]
                label = observation[-1]

                color_count = 0
                for color in COLORS:
                    if color_count >= 3:
                        break
                    # Extracting the pixel fraction for color
                    color_position = COLORS.index(color)
                    pixel_fraction = feature_list[color_position]

                    raw_values.append([color, pixel_fraction, label, frame_id])
                    color_count += 1
                
                frame_id += 1
        
            # DataFrame to contain the aggregate data
            df = pd.DataFrame(raw_values, columns=["color", "pixel_fraction", "label", "frame_id"])

            plt.close()
            sns.lineplot(data=df, x="frame_id", y="pixel_fraction", style="color", hue="label")
            #sns.scatterplot(data=df, x="frame_id", y="pixel_fraction", style="color", hue="label", size=1)
            plt.legend(ncol=4, loc="upper right")
            plt.savefig(join(outdir, "pixel_fraction_vs_%s_abs_count_%s.png"%(basename(video_file), str(absolute_pixel_count))), bbox_inches="tight")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzing the distribution of Pixel Fraction")
    parser.add_argument("-D", "--training-data", dest="training_data", help="Path to training data")
    parser.add_argument("-O", "--out-dir", dest="output_dir", help="Path to output data that would contain plots")
    
    args = parser.parse_args()
    main(args.training_data, args.output_dir)
