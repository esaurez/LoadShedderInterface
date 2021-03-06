import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
from os.path import join, basename
import os, sys
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../../"))
import model.build_model
import python_server.mapping_features

def main(bin_file, outdir, high_hue, low_hue, pf_cutoff):
    vid_name = basename(bin_file)[:-4]
    X = []
    cutoff_frames = []
    # Extracting the features from training data samples
    observations = python_server.mapping_features.read_samples(bin_file)
    positives = [[], []]
    negatives = [[], []]
    for frame_idx in range(len(observations)):
        frame = observations[frame_idx]
        label = frame.label
        hues = frame.feats.feats[0].feat.wholeHisto.histo.counts[0].count
        total_pixels = frame.feats.feats[0].feat.wholeHisto.histo.totalCountedPixels

        hue_pixels = 0
        for h in range(low_hue, high_hue+1):
            hue_pixels += hues[h]
        pf = hue_pixels/total_pixels
        if label:
            positives[0].append(frame_idx)
            positives[1].append(pf)
        else:
            negatives[0].append(frame_idx)
            negatives[1].append(pf)

        if pf_cutoff != None and pf >= pf_cutoff:
            cutoff_frames.append((frame_idx, pf))
    
    plt.scatter(positives[0], positives[1], label="+ve", s=1)
    plt.scatter(negatives[0], negatives[1], label="-ve", s=1)
    plt.legend()
    plt.xlabel("Frame ID")
    plt.ylabel("Pixel Fraction for Hue in [%d, %d]"%(low_hue, high_hue))
    plt.savefig(join(outdir, "hue_pf_%s_L_%d_H_%d.png"%(vid_name, low_hue, high_hue)), bbox_inches="tight")

    for (idx, pf) in cutoff_frames:
        print (idx, pf)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-B", dest="bin_file", help="Path to bin file")
    parser.add_argument("-O", dest="outdir", help="Output directory")
    parser.add_argument("-H", dest="high_hue", help="High cutoff for target hue", type=int)
    parser.add_argument("-L", dest="low_hue", help="Low cutoff for target hue", type=int)
    parser.add_argument("-C", dest="pf_cutoff", help="PF Cutoff", type=float)
    
    args = parser.parse_args()

    main(args.bin_file, args.outdir, args.high_hue, args.low_hue, args.pf_cutoff)
