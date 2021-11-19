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
from os import listdir
from os.path import basename, join, isfile, isdir

def get_uniq_obj_file(files, vid_name):
    for f in files:
        if vid_name in f:
            return f
    return None

def get_obj_times(uniq):
    first_line = True
    objs = []
    with open(uniq) as f:
        for l in f.readlines():
            if first_line:
                first_line = False
                continue

            s = l.split()
            start_end = []
            for idx in [1, 2]:
                t = s[idx]
                time_splits = t.split(":")
                secs = int(time_splits[0])*60 + int(time_splits[1])
                start_end.append(secs)
            objs.append(start_end)
    return objs

def main(util_csv, util_threshold, video_dir, fps):
    uniq_obj_files = [o for o in listdir(video_dir) if isfile(join(video_dir, o)) and o.endswith("_unique_objects.txt")]
    util_df = pd.read_csv(util_csv)
    video_name_grouping = util_df.groupby("video_name")

    aggr_total_objs = 0
    aggr_detected_objs = 0

    total_frames = 0
    frames_dropped = 0
    for vid in video_name_grouping.groups.keys():
        uniq_obj = get_uniq_obj_file(uniq_obj_files, vid)
        if uniq_obj:
            obj_times = get_obj_times(join(video_dir, uniq_obj))

            frames = []
            gdf = video_name_grouping.get_group(vid)
            for idx, row in gdf.iterrows():
                frames.append((row["frame_id"], row["label"], row["util"]))
            min_frame_idx = min([f[0] for f in frames])
            max_frame_idx = max([f[0] for f in frames])

            min_secs = min_frame_idx / fps
            max_secs = max_frame_idx / fps

            obj_covered = {}
            for obj_idx in range(len(obj_times)):
                s_e = obj_times[obj_idx]
                start = s_e[0]
                end = s_e[1]

                # Discarding all objects that were not covered in the utility file
                if end <= min_secs or start >= max_secs:
                    continue

                obj_found = False
                for (frame_idx, label, util) in frames:
                    if frame_idx/fps >= start and frame_idx/fps <= end and util >= util_threshold:
                        obj_found = True

                obj_covered[obj_idx] = obj_found

                for (frame_idx, label, util) in frames:
                    if util < util_threshold:
                        frames_dropped += 1
                    total_frames += 1
            
            if len(obj_covered) != 0:
                object_detection_rate = len([x for x in obj_covered if obj_covered[x] == True])/float(len(obj_covered))

                aggr_total_objs += len(obj_covered)
                aggr_detected_objs += len([x for x in obj_covered if obj_covered[x] == True])

    frame_drop_rate = frames_dropped/float(total_frames)

    print (float(aggr_detected_objs)/aggr_total_objs, frame_drop_rate)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzing the distribution of Pixel Fraction")
    parser.add_argument("-U", dest="util_csv", help="CSV containing the utility value for frames")
    parser.add_argument("-T", dest="util_threshold", type=float, help="Utility threshold")
    parser.add_argument("-I", dest="video_dir", help="Directory containing *_unique_objects.txt files")
    parser.add_argument("--fps", dest="fps", help="FPS of the bin file using with Util CSV has been calculated", type=int, required=True)

    args = parser.parse_args()
    main(args.util_csv, args.util_threshold, args.video_dir, args.fps)

