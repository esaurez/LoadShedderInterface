import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join

def aggregate(num_bins, positive_sv_mats, negative_sv_mats):
    max_val = 0
    sv_mats = {True: positive_sv_mats, False: negative_sv_mats}

    aggr = {}
    for label in sv_mats:
        aggr[label] = [[[] for col in range(num_bins)] for row in range(num_bins)]

    for label in sv_mats:
        for sv in sv_mats[label]:
            with open(sv) as f:
                row = 0
                for l in f.readlines():
                    split = l.split()
                    for col in range(len(split)):
                        aggr[label][row][col].append(float(split[col]))
                    row += 1 

        for row in range(num_bins):
            for col in range(num_bins):
                if row == 0 or col == 0:
                    aggr[label][row][col] = 0
                else:
                    aggr[label][row][col] = np.median(aggr[label][row][col])
                    max_val = max(aggr[label][row][col], max_val)
    return aggr, max_val

def main(color, num_bins, positive_sv_mats, negative_sv_mats, outdir):
    aggr, max_val = aggregate(num_bins, positive_sv_mats, negative_sv_mats)

    plt.close()
    cols = 2
    fig, axs = plt.subplots(1, cols, figsize=(cols*6, 1*4))
    labels = [True, False]
    fontsize = 16
    for label_idx in range(len(labels)):
        col = label_idx
        ax = axs[col]
        label = labels[label_idx]

        if label:
            label_str = "+ve frames"
        else:
            label_str = "-ve frames"

        sns.heatmap(aggr[label], ax=ax, cmap="BuPu", vmin=0, vmax=max_val)

        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.set_xlabel("Saturation bins", fontsize=fontsize)
        ax.set_ylabel("Value bins", fontsize=fontsize)
        plot_label = "%s ; %s"%(color.title(), label_str)
        ax.text(.5,.9, plot_label, horizontalalignment='center', transform=ax.transAxes, fontsize=fontsize)

    fig.savefig(join(outdir, "heatmap_%s_BINS_%d.png"%(color, num_bins)), bbox_inches="tight")

    # Dump the utility heatmap
    with open(join(outdir, "utils_%s_BINS_%d.txt"%(color, num_bins)), "w") as f:
        for row in range(num_bins):
            for col in range(num_bins):
                f.write("%f "%(aggr[True][row][col]))
            f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", dest="color", help="Color of SV mats that are being aggregated")
    parser.add_argument("-B", dest="num_bins", help="Number of bins in SV mats being aggregated", type=int)
    parser.add_argument("-P", dest="positive_sv_mats", help="SV Mats", nargs="+")
    parser.add_argument("-N", dest="negative_sv_mats", help="SV Mats", nargs="+")
    parser.add_argument("-O", dest="outdir", help="Output directory to store the utility matrix")
    args = parser.parse_args()

    main(args.color, args.num_bins, args.positive_sv_mats, args.negative_sv_mats, args.outdir)
