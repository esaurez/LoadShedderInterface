import argparse
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from os.path import join, isfile
from matplotlib.font_manager import FontProperties
import sys, os
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../"))

def main(frame_utils, outdir):
    df = pd.read_csv(frame_utils)

    plt.close()
    fig, ax = plt.subplots(figsize=(24,8))
    df.loc[df.composite_label == False, "composite_label"] = "-ve"
    df.loc[df.composite_label == True, "composite_label"] = "+ve"
    sns.boxplot(data=df, x="vid_name_x", y="composite_utility", hue="composite_label", ax=ax)
    num_vids = len(df["vid_name_x"].unique())
    fontsize = 20
    fontP = FontProperties()
    fontP.set_size(fontsize)
    ax.set_xlabel("Video file", fontsize=fontsize)
    ax.set_ylabel("Utility", fontsize=fontsize)
    ax.set_xticklabels(range(1, num_vids+1))
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    legend = ax.legend(fancybox=True,shadow=False,title='Frame label', prop=fontP, ncol=2)
    plt.setp(legend.get_title(),fontsize=fontsize)
    fig.savefig(join(outdir, "composite_cross_val__util_CROSSVIDEO.png"), bbox_inches="tight")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-U", dest="frame_utils", help="Path to frame_utils.csv generated by cross_validation")
    parser.add_argument("-O", dest="outdir", help="Output directory")

    args = parser.parse_args()

    main(args.frame_utils, args.outdir)
