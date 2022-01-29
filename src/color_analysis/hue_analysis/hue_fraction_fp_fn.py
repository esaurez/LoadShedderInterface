import argparse
import os, sys
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../../"))
sys.path.insert(0, os.path.join(script_dir, "../"))
import model.build_model
import python_server.mapping_features
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import glob
from os import listdir
from os.path import basename, join, isdir, isfile
import math
import numpy as np
import yaml
from capnp_serial import mapping_capnp
import object_based_metrics_calc

def compute_fp_fn(hf_map, hf_threshold):
    gt_pos = 0
    gt_neg = 0
    false_pos = 0
    false_neg = 0
   
    for vid in hf_map: 
        for (hf, label) in hf_map[vid]:
            if label:
                gt_pos += 1
                if hf < hf_threshold:
                    false_neg += 1
            else:
                gt_neg += 1
                if hf >= hf_threshold:
                    false_pos += 1
    
    fp = false_pos / gt_neg
    fn = false_neg / gt_pos
    return fp, fn

def extract_per_vid_hfs(bin_file, color, vidname):
    # Extracting the features from training data samples
    observations = python_server.mapping_features.read_samples(bin_file)
    frame_id=0
    labels = []
    hfs = []

    for observation in observations:
        if frame_id < 1:
            labels.append(False)
            hfs.append(0)
            frame_id += 1
            continue

        label = observation.label
        
        feature = observation.feats.feats[0]
        if feature.type == mapping_capnp.Feature.Type.fullColorHistogram:
            whole_histo = feature.feat.wholeHisto.histo
            hues = whole_histo.counts[0].count
            total_pixels = whole_histo.totalCountedPixels

            hue_count = 0
            for (low, high) in color:
                for idx in range(low, high):
                    hue_count += hues[idx]
            hf = hue_count/float(total_pixels) 
            labels.append(observation.label)
            hfs.append(hf)
        elif feature.type == mapping_capnp.Feature.Type.hsvHistogram:
            hists = feature.feat.hsvHisto.colorHistograms 
            fg_pixels = feature.feat.hsvHisto.totalCountedPixels
            if len(hists) > 1:
                print ("Multi color not supported rn. Exiting.")
                exit(1)
            hist = hists[0]
            
            total_color_pixels = 0
            for row_idx in range(len(hist.valueBins)):
                row = hist.valueBins[row_idx]
                for col_idx in range(len(row.counts)):
                    col = row.counts[col_idx]
                    total_color_pixels += col
            hue_fraction = total_color_pixels/fg_pixels
            labels.append(observation.label)
            hfs.append(hue_fraction)
        frame_id += 1
    
    df = pd.DataFrame.from_dict({"hf":hfs, "label":labels})
    df["vid"] = vidname
    hf_labels = [(hfs[idx], labels[idx]) for idx in range(len(labels))]
    return df, hf_labels, vidname

def compute_qor(hf_map, obj_frames, hf_threshold):
    total_frames = 0
    frames_dropped = 0
    total_objs = 0
    detected_objs = 0
    obj_sel_rates= []
    for vid in hf_map:
        objs = obj_frames[vid]
        is_frame_dropped = [hf < hf_threshold for (hf,l) in hf_map[vid]]
        obj_covered = object_based_metrics_calc.get_obj_coverage(objs, is_frame_dropped, 0)
        obj_frame_sel_rates = object_based_metrics_calc.get_obj_frame_sel_rates(objs, is_frame_dropped, 0)
        obj_sel_rates += obj_frame_sel_rates
        total_frames += len(is_frame_dropped)
        frames_dropped += len([x for x in is_frame_dropped if x])
        total_objs += len(obj_covered)
        detected_objs += len([x for x in obj_covered if obj_covered[x]])

    return detected_objs/total_objs, frames_dropped/total_frames, np.mean(obj_sel_rates) 

def main(training_conf_dir):
    with open(join(training_conf_dir, "conf.yaml")) as f:
        conf = yaml.safe_load(f)
    training_dir = conf["training_dir"]
    color = None
    for hue_bin in conf["hue_bins"]:
        name = hue_bin["name"]
        if color != None:
            print ("Number of colors has to be 1 in this experiment. Exiting")
            exit(1)
        color = []
        for r in hue_bin["ranges"]:
            color.append((r["start"], r["end"]))

    executor = ProcessPoolExecutor(max_workers=8)
    obj_frames = {}
    vids = [d for d in listdir(training_dir) if isdir(join(training_dir, d))]
    futures = []
    for vid in vids:
        vid_dir = join(training_dir, vid)
        bin_file = [join(vid_dir, f) for f in listdir(vid_dir) if isfile(join(vid_dir, f)) and f.endswith(".bin")][0]

        future = executor.submit(extract_per_vid_hfs, bin_file, color, vid)

        futures.append(future)

        uniq_obj_file = join(join(training_dir, vid), "unique_objs_per_frame.txt")
        if not isfile(uniq_obj_file):
            continue
        obj_frames[vid] = object_based_metrics_calc.get_obj_frames(uniq_obj_file)

    dfs = []
    hf_map = {}
    max_hf = None
    for f in futures:
        df, hfs, vid = f.result()
        hf_map[vid] = hfs
        dfs.append(df)
        M = max([hf for (hf, l) in hfs])
        if max_hf == None or M> max_hf:
            max_hf= M

    df = pd.concat(dfs, ignore_index=True)

    fontsize=20
    fontP = FontProperties()
    fontP.set_size(fontsize)

    df.loc[df.label == False, "label"] = "-ve"
    df.loc[df.label == True, "label"] = "+ve"
    num_vids = len(df["vid"].unique())

    plt.close()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.stripplot(data=df, x="vid", y="hf", hue="label", dodge=True, ax=ax)
    ax.set_xlabel("Video file")
    ax.set_ylabel("Hue Fraction")
    ax.grid()
    fig.savefig("hf_stripplot.png", bbox_inches="tight")
    plt.close()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.boxplot(data=df, x="vid", y="hf", hue="label", ax=ax)
    ax.set_xlabel("Video File", fontsize=fontsize)
    ax.set_ylabel("Hue Fraction", fontsize=fontsize)
    ax.grid()
    ax.set_xticklabels(range(1, num_vids+1))
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    legend = ax.legend(fancybox=True,shadow=False,title='Frame label', prop=fontP, ncol=2)
    plt.setp(legend.get_title(),fontsize=fontsize)
    fig.savefig("hf_catplot.png", bbox_inches="tight")
    
    num_points = 100
    step = max_hf/num_points
    X = []
    FP = []
    FN = []

    obj_det_rates = []
    qors = []
    frame_drops = []

    for idx in range(num_points):
        hf_threshold = idx*step
        #fp, fn = compute_fp_fn(hf_map, hf_threshold)
        obj_det_rate, frame_drop_rate, qor = compute_qor(hf_map, obj_frames, hf_threshold)
        X.append(hf_threshold)
        #FP.append(fp)
        #FN.append(fn)
        frame_drops.append(frame_drop_rate)
        qors.append(qor)
        obj_det_rates.append(obj_det_rate)
    plt.close()
    fig, ax = plt.subplots(figsize=(8,6))
    #ax.plot(X, FP, label="FP")
    #ax.plot(X, FN, label="FN")
    #ax.plot(X, obj_det_rates, label="Obj. Detn. Rate")
    ax.plot(X, qors, label="Per-Object QoR")
    ax.plot(X, frame_drops, label="Frame Drop Rate")
    ax.set_xlabel("Hue Fraction threshold", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    legend = ax.legend(fancybox=True,shadow=False, prop=fontP)

    ax.grid()
    fig.savefig("hue_fp_np.png", bbox_inches="tight")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzing what should be the cutoff Hue Fraction.")
    parser.add_argument("-C", dest="training_conf_dir", help="Path to training conf folder")

    args = parser.parse_args()
    main(args.training_conf_dir)

