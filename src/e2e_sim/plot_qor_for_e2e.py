import argparse
import random
import numpy as np
from os import listdir
from os.path import join, isfile, isdir
import yaml
import pandas as pd
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../color_analysis"))
sys.path.insert(0, os.path.join(script_dir, "../color_analysis/sv_matrix"))
import object_based_metrics_calc
sys.path.insert(0, os.path.join(script_dir, "../"))
import model.build_model
import python_server.mapping_features

def main(training_conf, shedding_decisions_dir, dnn_ms, outdir, fps):
    conf_file = join(training_conf, "conf.yaml")
    with open(conf_file) as f:
        conf = yaml.safe_load(f)
    training_dir = conf["training_dir"]
    training_split = conf["training_split"]

    vids = [v for v in listdir(training_dir) if isdir(join(training_dir, v))]
    uniq_objs = {}
    num_frames = {}
    for vid in vids:
        vid_dir = join(training_dir, vid)
        uniq_objs[vid] = object_based_metrics_calc.get_obj_frames(join(vid_dir, "unique_objs_per_frame.txt"))
        # Get total length of video in terms of frames from the bin file
        bin_file = [join(vid_dir, f) for f in listdir(vid_dir) if isfile(join(vid_dir, f)) and f.endswith(".bin")][0]
        observations = python_server.mapping_features.read_samples(bin_file)
        num_frames[vid] = len(observations)

    shedding_files = [join(shedding_decisions_dir, f) for f in listdir(shedding_decisions_dir) if isfile(join(shedding_decisions_dir, f)) and f.startswith("shedding_decisions") and f.endswith(".csv")]

    dfs = []
    for shedding_decisions_file in shedding_files:
        df = pd.read_csv(shedding_decisions_file)

        # Now the per video is_frame_dropped array
        grouping = df.groupby("video_name")
        num_vids = len(grouping.groups.keys())

        #if num_vids % 2 == 0:
        #    continue

        # Calculate target drop rate for random shedding
        total_incoming_fps = fps*num_vids
        sustained_tput = 1000.0/dnn_ms
        target_drop_rate = max(0, 1 - sustained_tput/total_incoming_fps)

        sel_rates = []
        sel_rates_rand = []
        for vid in grouping.groups.keys():
            gdf = grouping.get_group(vid)
            decisions = []
            for idx, row in gdf.iterrows():
                shed_decision = row["shed_decision"]
                frame_idx = row["frame_id"]
                decisions.append((frame_idx, shed_decision))
            decisions = sorted(decisions, key = lambda x : x[0])
            is_frame_dropped = [None for idx in range(num_frames[vid])]
            for (idx, decision) in decisions:
                is_frame_dropped[idx] = not decision        

            # Now calculate the object based metric for util based approach
            rates = object_based_metrics_calc.get_obj_frame_sel_rates(uniq_objs[vid], is_frame_dropped, 0)
            sel_rates += rates

            # Calculate QoR for random shedding approach
            random.seed(vid+shedding_decisions_file)
            is_frame_dropped_rand = [random.uniform(0,1) < target_drop_rate for idx in range(num_frames[vid])]
            rand_rates = object_based_metrics_calc.get_obj_frame_sel_rates(uniq_objs[vid], is_frame_dropped_rand, 0)
            sel_rates_rand += rand_rates
   
        qor = [s for s in sel_rates]+[s for s in sel_rates_rand]
        shedder = ["Utility-based" for s in sel_rates]+["Content-agnostic" for s in sel_rates_rand]
 
        sel_df = pd.DataFrame.from_dict({"qor":qor, "shedder":shedder})
        sel_df["num_vids"] = num_vids
        dfs.append(sel_df)

    df = pd.concat(dfs, ignore_index=False)

    plt.close()
    fontsize=20
    fig, ax = plt.subplots(figsize=(8,4))
    sns.boxplot(data = df, x="num_vids", y="qor", hue="shedder", ax=ax)
    fontP = FontProperties()
    fontP.set_size(fontsize)
    ax.legend(prop=fontP, loc="right")
    ax.set_xlabel("Number of concurrent video streams", fontsize=fontsize)
    ax.set_ylabel("Per-object QoR metric", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    fig.savefig(join(outdir, "qor_vs_num_vids_D_%d_FPS_%d.png"%(dnn_ms, fps)), bbox_inches="tight")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", dest="training_conf", help="Path to training conf dir", required=True)
    parser.add_argument("-I", dest="shedding_decisions_dir", help="Path to file containing the shedding_decisions_*.csv", required=True)
    parser.add_argument("-D", dest="dnn_inference_latency_ms", help="Latency of executing the backend query", type=int)
    parser.add_argument("--fps", dest="fps", help="FPS of each camera's video", type=int)
    parser.add_argument("-O", dest="outdir", help="Output directory", default="tmp")
    args = parser.parse_args()
    main(args.training_conf, args.shedding_decisions_dir, args.dnn_inference_latency_ms, args.outdir, args.fps)
