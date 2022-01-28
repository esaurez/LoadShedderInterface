import argparse
import yaml
import os, sys
from os import listdir
from os.path import join, isfile, basename
from concurrent.futures import ProcessPoolExecutor
import concurrent
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../../../"))
import python_server.mapping_features
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

max_frames=None

def map_frame_to_conf_color_idx(frame_colors, conf_colors):
    mapping = {}
    frame_color_idx = 0
    for color in frame_colors:
        hue_ranges = []
        for r in color.ranges:
            hue_ranges.append((r.hueBegin, r.hueEnd))
        hue_ranges = sorted(hue_ranges, key = lambda x: x[0])
        for conf_idx in range(len(conf_colors)):
            conf_color = conf_colors[conf_idx]
            if len(conf_color) != len(hue_ranges):
                continue
            for range_idx in range(len(conf_color)):
                if conf_color[range_idx] != hue_ranges[range_idx]:
                    continue
            mapping[frame_color_idx] = conf_idx
            break
        frame_color_idx += 1
    return mapping

def get_sv_counts(frame, conf_colors, num_bins, pf_threshold):
    frame_colors = frame.feats.feats[0].feat.hsvHisto.colorRanges
    hists = frame.feats.feats[0].feat.hsvHisto.colorHistograms
    frame_to_conf = map_frame_to_conf_color_idx(frame_colors, conf_colors)

    sv_mats = {}
    total_pixels = {}
    total_foreground_pixels = frame.feats.feats[0].feat.hsvHisto.totalCountedPixels
    
    for idx in range(len(frame_colors)):
        conf_color_idx = frame_to_conf[idx]
        sv_mats[conf_color_idx] = []
        hist = hists[idx]

        total_color_pixels = 0
        total_counted_pixels = 0
        for row_idx in range(len(hist.valueBins)):
            row = hist.valueBins[row_idx]
            sv_mats[conf_color_idx].append([])
            if len(sv_mats[conf_color_idx][-1]) == 0:
                for col_idx in range(len(row.counts)):
                    sv_mats[conf_color_idx][-1].append(0)
                    
            for col_idx in range(len(row.counts)):
                col = row.counts[col_idx]
                total_color_pixels += col
                #TODO
                #if row_idx == 0 or col_idx == 0:
                #    continue
                sv_mats[conf_color_idx][row_idx][col_idx] += col
                total_counted_pixels += col
        
        if total_counted_pixels > 0:
            for row_idx in range(len(sv_mats[conf_color_idx])):
                for col_idx in range(len(sv_mats[conf_color_idx][row_idx])):
                    sv_mats[conf_color_idx][row_idx][col_idx] /= total_counted_pixels

        hue_fraction = total_color_pixels/total_foreground_pixels
        if hue_fraction < pf_threshold:
            sv_mats[conf_color_idx] = None

    return sv_mats

def dump_sv_mat(mat, outfile):
    with open(outfile, "w") as f:
        for row in mat:
            for col in row:
                f.write("%f "%col)
            f.write("\n")

def get_conf_colors(training_conf):
    conf_colors = []
    for color in training_conf["hue_bins"]:
        color_name = color["name"]
        ranges = color["ranges"]
        range_limits = []
        for hue_range in ranges:
            r = (hue_range["start"], hue_range["end"])
            range_limits.append(r)
        range_limits = sorted(range_limits, key = lambda x : x[0])
        conf_colors.append(range_limits)
    return conf_colors

def main(training_conf_file, num_bins, bin_file, outdir, pf_threshold, training_split):
    with open(training_conf_file) as f:
        training_conf = yaml.safe_load(f)
   
    if training_split == None:
        training_split = training_conf["training_split"]

    conf_colors = get_conf_colors(training_conf)
    # Reading information about the ground-truth of frames from the bin file
    ground_truth_frames = python_server.mapping_features.read_samples(bin_file)
    # Creating the ProcessPoolExecutor
    executor = ProcessPoolExecutor(max_workers=32)

    colors = training_conf["hue_bins"]
    sv_mat_list = [[] for idx in range(len(colors))] # Initially designed to work for multiple colors. Right now only 1 color (w/ multiple hue ranges) is used 

    num_total_frames = len(ground_truth_frames)
    num_training_frames = int(training_split*num_total_frames)

    futures = []
    for frame_idx in range(num_total_frames):
        if frame_idx >= num_training_frames:
            break
        frame = ground_truth_frames[frame_idx]

        # Submitting a parallel job for each frame
        future = executor.submit(get_sv_counts, frame, conf_colors, num_bins, pf_threshold) # function returns none if the pixel fraction is too low
        futures.append(future)

        if max_frames != None and frame_idx > max_frames:
            break

    # Reading the result of each parallel job
    for f in futures:
        mats_list = f.result()
        for color_idx in range(len(mats_list)):
            # The result of get_sv_counts contains a list with 1 element per color
            sv_mat_list[color_idx].append(mats_list[color_idx])

    for x in sv_mat_list:
        print (len(x))

    # Now aggregate the sv_mats over frames and labels
    aggr_list_sv_mats = [{} for c in range(len(colors))]
    aggr_sv_mats = [{} for c in range(len(colors))]
    for color_idx in range(len(colors)):
        for label in [True, False]:
            aggr_list_sv_mats[color_idx][label] = [[[] for col in range(num_bins)] for row in range(num_bins)]
            aggr_sv_mats[color_idx][label] = [[0 for col in range(num_bins)] for row in range(num_bins)]
   
    # TODO NUM BINS AND BIN SIZE SHOULD NOT BE EQUAL 
 
    # First creating a list of per-frame matrices
    for frame_idx in range(len(ground_truth_frames)):
        if frame_idx >= num_training_frames:
            break
        if max_frames != None and frame_idx > max_frames:
            break
        label = ground_truth_frames[frame_idx].label
        for color_idx in range(len(colors)):
            frame_mat = sv_mat_list[color_idx][frame_idx]
            if frame_mat == None: # None due to very low PF
                continue
            for row in range(len(frame_mat)):
                for col in range(len(frame_mat[row])):
                    if frame_mat[row][col] > 0:
                        aggr_list_sv_mats[color_idx][label][row][col].append(frame_mat[row][col])
   
    # Aggregating the list of per-frame matrices by taking average over each matrix cell
    for color_idx in range(len(colors)):
        for label in [True, False]:
            for row in range(len(aggr_sv_mats[color_idx][label])):
                for col in range(len(aggr_sv_mats[color_idx][label][row])):
                    if len(aggr_list_sv_mats[color_idx][label][row][col]) == 0:
                        aggr_sv_mats[color_idx][label][row][col] = 0
                    else:
                        aggr_sv_mats[color_idx][label][row][col] = np.mean(aggr_list_sv_mats[color_idx][label][row][col])

    for color_idx in range(len(colors)):
        color_name = colors[color_idx]["name"]
        for label in [True, False]:
            outfile = join(outdir, "sv_matrix_label_%s_BINS_%d_C_%s.txt"%(str(label), num_bins, color_name))
            dump_sv_mat(aggr_sv_mats[color_idx][label], outfile)

    plt.close()
    rows = len(colors)
    cols = 2
    fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    idx = 0
    for color_idx in range(len(colors)):
        row = color_idx
        labels = [True, False]
        for label_idx in range(len(labels)):
            col = label_idx
            if rows  == 1:
                ax = axs[col]
            else:
                ax = axs[row][col]
            label = labels[label_idx]

            if label:
                label_str = "+ve frames"
            else:
                label_str = "-ve frames"

            sns.heatmap(aggr_sv_mats[color_idx][label], ax=ax, cmap="BuPu", vmin=0)

            plot_label = "%s ; %s"%(str(colors[color_idx]["name"]), label_str)
            ax.text(.5,.9, plot_label, horizontalalignment='center', transform=ax.transAxes)
            idx += 1

    heatmap_filename = basename(bin_file)[:-4]
    fig.savefig(join(outdir, "heatmap_%s_BINS_%d.png"%(heatmap_filename, num_bins)), bbox_inches="tight")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzing the distribution of Pixel Fraction")
    parser.add_argument("-B", dest="bin_file", help="Path to bin file")
    parser.add_argument("-C", dest="training_conf", help="Path to the training conf yaml")
    parser.add_argument("-O", dest="outdir", help="Path to output directory")
    parser.add_argument("--bins", dest="num_bins", help="Number of bins", type=int, default=16)
    parser.add_argument("--pf-threshold", dest="pf_threshold", help="Min fraction of pixels (with sat>0 or val>0) for frame to be considered for processing.", type=float, default=0.025)
    parser.add_argument("--training-split", dest="training_split", help="Training split override.", type=float)

    args = parser.parse_args()
    main(args.training_conf, args.num_bins, args.bin_file, args.outdir, args.pf_threshold, args.training_split)

