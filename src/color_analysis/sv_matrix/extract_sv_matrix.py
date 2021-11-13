import argparse
import yaml
import os, sys
from os import listdir
from os.path import join, isfile, basename
from concurrent.futures import ProcessPoolExecutor
import concurrent
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../../"))
import python_server.mapping_features
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

max_frames=None

def get_sv_counts(frame, colors, bin_size):
    sv_mats = {}
    total_pixels = {}
    for color_idx in range(len(colors)):
        sv_mats[color_idx] = [[0 for col in range(bin_size)] for row in range(bin_size)]
        total_pixels[color_idx] = 0

    with open(frame) as f:
        for line in f.readlines():
            s = line.split()
            hue = int(s[0])
            sat = int(s[1])
            val = int(s[2])

            if sat == 0 or val == 0:
                continue

            for color_idx in range(len(colors)):
                color = colors[color_idx]
                pixel_matches = False
                for hue_range in color["ranges"]:
                    if hue >= hue_range["start"] and hue <= hue_range["end"]:
                        pixel_matches = True
                        break
                if pixel_matches:
                    sat_idx = int(sat/bin_size)
                    val_idx = int(val/bin_size)
                    if sat_idx <1 or val_idx < 1:
                        continue
                    sv_mats[color_idx][val_idx][sat_idx] += 1
                    total_pixels[color_idx] += 1

    # Now normalize
    for color_idx in range(len(colors)):
        for row in range(bin_size):
            for col in range(bin_size):
                if total_pixels[color_idx] == 0:
                    sv_mats[color_idx][row][col] = 0
                else:
                    sv_mats[color_idx][row][col] /= float(total_pixels[color_idx])

    return sv_mats

def main(frame_dir, training_conf_file, bin_size, bin_file, outdir):
    ground_truth_frames = python_server.mapping_features.read_samples(bin_file)

    with open(training_conf_file) as f:
        training_conf = yaml.load(f)
   
    # Creating the ProcessPoolExecutor
    executor = ProcessPoolExecutor(max_workers=4)

    colors = training_conf["hue_bins"]
    sv_mat_list = [[] for idx in range(len(colors))]

    # Assign a unique index to each
    frames = [join(frame_dir, f) for f in listdir(frame_dir) if isfile(join(frame_dir, f))]
    frames = sorted(frames, key = lambda x : int(basename(x)[:-4].split("_")[1]))

    futures = []
    for frame_idx in range(len(frames)):
        frame = frames[frame_idx]
        future = executor.submit(get_sv_counts, frame, colors, bin_size)
        futures.append(future)

        if max_frames != None and frame_idx > max_frames:
            break

    for f in futures:
        mats_list = f.result()
        for color_idx in range(len(mats_list)):
            sv_mat_list[color_idx].append(mats_list[color_idx])

    for x in sv_mat_list:
        print (len(x))

    # Now aggregate the sv_mats over frames and labels
    aggr_list_sv_mats = [{} for c in range(len(colors))]
    aggr_sv_mats = [{} for c in range(len(colors))]
    for color_idx in range(len(colors)):
        for label in [True, False]:
            aggr_list_sv_mats[color_idx][label] = [[[] for col in range(bin_size)] for row in range(bin_size)]
            aggr_sv_mats[color_idx][label] = [[0 for col in range(bin_size)] for row in range(bin_size)]
   
    # TODO NUM BINS AND BIN SIZE SHOULD NOT BE EQUAL 
 
    for frame_idx in range(len(ground_truth_frames)):
        if max_frames != None and frame_idx > max_frames:
            break
        label = ground_truth_frames[frame_idx].label
        for color_idx in range(len(colors)):
            frame_mat = sv_mat_list[color_idx][frame_idx]
            for row in range(len(frame_mat)):
                for col in range(len(frame_mat[row])):
                    if frame_mat[row][col] > 0:
                        aggr_list_sv_mats[color_idx][label][row][col].append(frame_mat[row][col])
   
    #for row in aggr_list_sv_mats[0][True]:
    #    for x in row:
    #        print (np.mean(x), " ", end="")
    #    print ("")
 
    for color_idx in range(len(colors)):
        for label in [True, False]:
            for row in range(len(aggr_sv_mats[color_idx][label])):
                for col in range(len(aggr_sv_mats[color_idx][label][row])):
                    if len(aggr_list_sv_mats[color_idx][label][row][col]) == 0:
                        aggr_sv_mats[color_idx][label][row][col] = 0
                    else:
                        aggr_sv_mats[color_idx][label][row][col] = np.mean(aggr_list_sv_mats[color_idx][label][row][col])

    #print (aggr_sv_mats[0][True])

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
    fig.savefig(join(outdir, "heatmap_%s.png"%heatmap_filename), bbox_inches="tight")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzing the distribution of Pixel Fraction")
    parser.add_argument("-F", dest="frame_dir", help="Path to directory containing hsv for each frame")
    parser.add_argument("-B", dest="bin_file", help="Path to bin file")
    parser.add_argument("-C", dest="training_conf", help="Path to the training conf yaml")
    parser.add_argument("-O", dest="outdir", help="Path to output directory")

    args = parser.parse_args()
    main(args.frame_dir, args.training_conf, 16, args.bin_file, args.outdir)

