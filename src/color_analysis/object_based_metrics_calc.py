import argparse
import yaml
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

def get_obj_frames(uniq):
    first_line = True
    result = {} # objID --> [list of frameIDs]
    with open(uniq) as f:
        for l in f.readlines():
            if first_line:
                first_line = False
                continue

            s = l.split()

            fidx = int(s[0])
            objs = [int(s[i]) for i in range(1, len(s))]

            for obj in objs:
                if obj not in result:
                    result[obj] = []
                result[obj].append(fidx)
    return result

def get_obj_coverage(obj_frames, frames_dropped, num_training_frames):
    obj_covered = {}
    for obj in obj_frames:
        obj_in_training_data = False
        obj_found = False
        frames = obj_frames[obj]

        for fidx in frames:
            if fidx < num_training_frames:
                obj_in_training_data = True

            if not frames_dropped[fidx]:
                obj_found = True

        if not obj_in_training_data:
            global_obj_id = str(obj)
            obj_covered[global_obj_id] = obj_found

    return obj_covered

def main(training_conf, util_threshold):
    with open(join(training_conf, "conf.yaml")) as f:
        conf = yaml.safe_load(f)

    training_dir = conf["training_dir"]
    training_split = conf["training_split"]
    vid_dirs = [join(training_dir, d) for d in listdir(training_dir) if isdir(join(training_dir, d))]

    aggr_total_objs = 0
    aggr_detected_objs = 0

    total_frames = 0
    frames_dropped = 0
    for vid_dir in vid_dirs:
        uniq_obj = join(vid_dir,  "unique_objs_per_frame.txt")
        if not isfile(uniq_obj):
            continue
        util_df = pd.read_csv(join(vid_dir, "frame_utils.csv"))
        num_frames = len(util_df)
        num_training_frames = int(num_frames*training_split)
        utils = []
        for idx , row in util_df.iterrows():
            utils.append(row["utility"])

        total_frames += len(utils[:num_training_frames])
        frames_dropped += len([f for f in utils[:num_training_frames] if f < util_threshold])

        obj_frames = get_obj_frames(join(vid_dir, uniq_obj))
        
        is_frame_dropped = []
        for idx in range(len(utils)):
            if utils[idx] >= util_threshold:
                is_frame_dropped.append(False)
            else:
                is_frame_dropped.append(True)
        obj_covered = get_obj_coverage(obj_frames, is_frame_dropped, num_training_frames)

        if len(obj_covered) != 0:
            object_detection_rate = len([x for x in obj_covered if obj_covered[x] == True])/float(len(obj_covered))
    
            aggr_total_objs += len(obj_covered)
            aggr_detected_objs += len([x for x in obj_covered if obj_covered[x] == True])

    frame_drop_rate = frames_dropped/float(total_frames)

    print ("%.5f\t%.3f\t%.3f"%(util_threshold, float(aggr_detected_objs)/aggr_total_objs, frame_drop_rate))
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzing the distribution of Pixel Fraction")
    parser.add_argument("-T", dest="util_threshold", type=float, help="Utility threshold")
    parser.add_argument("-C", dest="training_conf", help="Directory containing Training Conf")

    args = parser.parse_args()

    util_thresholds = []
    u = 0
    while u <= 0.04:
        util_thresholds.append(u)
        main(args.training_conf, u)
        u += 0.0005


