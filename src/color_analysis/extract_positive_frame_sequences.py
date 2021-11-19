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
import math

def get_time(idx, fps):
    frame_sec = idx/fps
    # Now convert frame_sec to minutes
    mins = int(frame_sec/60)
    secs = frame_sec - mins*60
    return "%d:%.2f"%(mins, secs)

def sequences_too_close(prev, curr, fps):
    if curr[0] - prev[1] <= 5/fps: # and curr[2] == prev[2]:
        return True
    else:
        return False

def main(bin_file, fps):
    video_file = bin_file
    raw_values = [] # array to contain the raw values which will later be converted to dataframe

    video_file_idx = 0
    # Extracting the features from training data samples
    observations = python_server.mapping_features.read_samples(video_file)
    #observations = python_server.mapping_features.read_training_file(video_file)
    frame_id=0
    labels = []

    sequences = [] # inclusive ranges

    for observation in observations:
            label = 0
            if observation.label:
                label = 1
                labels.append((label, observation.detections.totalDetections))
            else:
                label = 0
                labels.append((label, 0))

    curr_seq_start = None
    curr_detections = 0
    for idx in range(len(labels)):
        l, count = labels[idx]
        curr_detections = max(curr_detections, count)
        if l == 0:
            if curr_seq_start != None:
                # Sequence ending
                curr_seq = (curr_seq_start, idx-1, curr_detections)
                if len(sequences) > 0 and sequences_too_close(sequences[-1], curr_seq, fps):
                    sequences[-1] = (sequences[-1][0], curr_seq[1], max(sequences[-1][2], curr_seq[2]))
                else:
                    sequences.append(curr_seq)
                curr_seq_start = None
                curr_detections = 0
        else:
            if curr_seq_start == None:
                curr_seq_start = idx
                curr_detections = count

    # End of the frames
    if curr_seq_start != None:
        curr_seq = (curr_seq_start, idx-1, curr_detections)
        if len(sequences) > 0 and sequences_too_close(sequences[-1], curr_seq, fps):
            sequences[-1] = (sequences[-1][0], curr_seq[1], max(sequences[-1][2], curr_seq[2]))
        else:
            sequences.append(curr_seq)

    for (s, e, count) in sequences:
        print (get_time(s, fps), "\t", get_time(e, fps), "\t", count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzing the distribution of Pixel Fraction")
    parser.add_argument("-B", "--bin-file", dest="bin_file", help="BIN file")
    parser.add_argument("-F", "--fps", dest="fps", help="FPS of the bin file", type=float, required=True)
    
    args = parser.parse_args()
    main(args.bin_file, args.fps)
