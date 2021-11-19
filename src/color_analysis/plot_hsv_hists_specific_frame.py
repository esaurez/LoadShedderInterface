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

def get_pdf_xy(counts):
    X = []
    Y = []
    cum_sum = 0
    for idx in range(len(counts)):
        X.append(idx)
        Y.append(counts[idx])
    return (X, Y)

def plot_hsv_hists(frame, outfile):
    counts = frame.feats.feats[0].feat.wholeHisto.histo.counts
    hue = counts[0].count
    saturation = counts[1].count
    value = counts[2].count

    counts = [hue, saturation, value]
    labels = ["Hue", "Saturation", "Value"]
    pdfs = [get_pdf_xy(c) for c in counts]

    plt.close()
    idx = 0
    for pdf in pdfs:
        plt.plot(pdf[0], pdf[1], label=labels[idx])
        idx += 1
    plt.legend()
    plt.savefig(outfile, bbox_inches="tight")

def main(video_file, target_frame_id, outdir):
    frame_id = 0
    # Extracting the features from training data samples
    training_samples = python_server.mapping_features.read_samples(video_file)

    outplot = join(outdir, "hsv_V_%s_F_%d.png"%(basename(video_file), target_frame_id))

    for sample in training_samples:
        if frame_id == target_frame_id:
            plot_hsv_hists(sample, outplot)
            return
        frame_id += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzing the distribution of Pixel Fraction")
    parser.add_argument("-V", "--video-file", dest="video_file", help="Path to video file")
    parser.add_argument("-F", "--frame-id", type=int, dest="frame_id", help="ID of the frame (starts from 0)")
    parser.add_argument("-O", "--outdir", dest="outdir", help="Output directory to store the plots")
    
    args = parser.parse_args()
    main(args.video_file, args.frame_id, args.outdir)
