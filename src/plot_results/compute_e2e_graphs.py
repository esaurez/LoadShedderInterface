#!/usr/bin/env python3

import numpy as np
import argparse
import csv
from enum import Enum
import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns
import sys

class Levels(Enum):
    SHEDDER = 1 
    FILTER = 2
    DETECTION = 3 
    DETECTION_FILTER = 4
    SINK = 5

level_column = {Levels.SHEDDER: "Shedder", Levels.FILTER: "Filter", Levels.DETECTION: "Detection", Levels.DETECTION_FILTER: "Detection Filter", Levels.SINK: "Sink"}

level_order = [level_column[Levels.SHEDDER],level_column[Levels.FILTER],level_column[Levels.DETECTION],level_column[Levels.DETECTION_FILTER],level_column[Levels.SINK]]

def plot_e2e_latency(dataframe, axes):
    ax = sns.lineplot(data=dataframe, x="Frame ID", y="Total Latency [ms]", ax=axes)
    #ax.set_ylim(0, 120.0)
    #ax.set_xlim(0, float(x_max_lim))

def plot_e2e_types(dataframe, axes):
    #ax = sns.stripplot(data=dataframe, x="Frame ID", y="level", order=level_order, jitter=False, ax=axes)
    ax = sns.stripplot(data=dataframe, x="Frame ID", y="End Node", ax=axes)
    #ax.set_ylim(0, 120.0)
    #ax.set_xlim(0, float(x_max_lim))
def plot_e2e(dataframe): 
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(6,10)) 
    #figure.suptitle('Big title for plot')    
    plot_e2e_latency(dataframe, axes[0])
    plot_e2e_types(dataframe, axes[1])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Frame Id", size=24)
    #plt.ylabel("Average\nSpatial Alignment [%]", size=24)
    plt.savefig("e2e_latency_location.pdf", bbox_inches="tight")
    plt.close()

def getUsefulValues(row):
    #video,segment_id,frame_id,threshold,utility,shed_decision,useful_gt,shedder_begin,shedder_end,filter_begin,filter_end,detection_begin,detection_end,det_filter_begin,det_filter_end,de     t_sink_begin,det_sink_end
    threshold = row['threshold']
    utility = row['utility']
    if not row["filter_begin"]:
        latency = (int(row["shedder_end"]) - int(row["shedder_begin"]))/1000000
        level = level_column[Levels.SHEDDER]
    elif not row["detection_begin"]:
        latency = (int(row["filter_end"]) - int(row["shedder_begin"]))/1000000
        level = level_column[Levels.FILTER]
    elif not row["det_filter_begin"]:
        latency = (int(row["detection_end"]) - int(row["shedder_begin"]))/1000000
        level = level_column[Levels.DETECTION]
    elif not row["det_sink_begin"]:
        latency = (int(row["det_filter_end"]) - int(row["shedder_begin"]))/1000000
        level = level_column[Levels.DETECTION_FILTER]
    else:
        latency = (int(row["det_sink_end"]) - int(row["shedder_begin"]))/1000000
        level = level_column[Levels.SINK]
    return latency,level,threshold,utility

def get_dataframe(csv_file, video):
    with open(csv_file) as csvFile:
        reader = csv.DictReader(csvFile)
        # The headers are as follow
        data = []
        for row in reader:
            if video == row['video']:
                latency,level, threshold, utility  = getUsefulValues(row)
                data.append([int(row["frame_id"]),latency, level, threshold, utility, row["shed_decision"]])
        return DataFrame(data, columns=["Frame ID","Total Latency [ms]", "End Node", "Threshold", "Utility","Shed Decision"])

def main(csv_file, video):
    palette1 = sns.color_palette("colorblind", 8)
    sns.set_palette(palette1)
    sns.set(font_scale = 1.47)
    sns.set_style("whitegrid")
    dataframe = get_dataframe(csv_file, video)
    plot_e2e(dataframe)
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-F", dest = "csv_file", type=str, help="File that contains the csv file with the results of a run",required=True)
    parser.add_argument("-V", dest = "video", type=str, help="Name of video",required=True)
    args = parser.parse_args()
    main(args.csv_file, args.video)
