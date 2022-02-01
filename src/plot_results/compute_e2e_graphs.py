#!/usr/bin/env python3

import numpy as np
import argparse
import csv
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
from pandas import DataFrame
import seaborn as sns
import sys
import random

def getVideoName(videoIdx):
    if videoIdx == 0:
        return "seed949563-3-train"
    elif videoIdx == 1:
        return "seed949563-1-train"
    elif videoIdx == 2:
        return "seed949563-0-train"
    elif videoIdx == 3:
        return "seed10484-3-train"
    elif videoIdx == 4:
        return "seed949563-2-train"
    elif videoIdx == 5:
        return "seed10484-1-train"
    elif videoIdx == 6:
        return "seed335500-0-train"
    raise Exception("Invalid videoIdx ")

class Levels(Enum):
    SHEDDER = 1 
    FILTER = 2
    DETECTION = 3 
    DETECTION_FILTER = 4
    SINK = 5

level_column = {Levels.SHEDDER: "Shedder", Levels.FILTER: "Filter", Levels.DETECTION: "Detection", Levels.DETECTION_FILTER: "Detection Filter", Levels.SINK: "Sink"}

level_order = [level_column[Levels.SHEDDER],level_column[Levels.FILTER],level_column[Levels.DETECTION],level_column[Levels.DETECTION_FILTER],level_column[Levels.SINK]]

def divide(x):
    FPS=10
    SECONDS=5
    division = FPS * SECONDS
    return int(int(x)/int(division))*SECONDS

def plot_e2e_latency(dataframe, axes, threshold, vertical_lines):
    dataframe["Seconds"]=dataframe["Frame ID"].apply(divide)
    dataframe = dataframe.loc[dataframe['Total Latency [ms]'] > 0]
    #print(dataframe.loc[dataframe['Total Latency [ms]'] > 0].to_string())
    df = dataframe.groupby(["Seconds"]).max()#.quantile(0.99)#.quantile(0.99)
    yaxis_name = "Average Latency [ms] \n per 5 secs bucket"
    df[yaxis_name] = df["Total Latency [ms]"]
    ax = sns.lineplot(data=df, x="Seconds", y=yaxis_name, ax=axes, linewidth = 2.5)
    axes.axhline(y=threshold, color='firebrick', linestyle='--',  label='Latency Requirement',linewidth=2.5)
    axes.text(340,threshold + 50,'Latency Requirement',rotation=360, color='firebrick')
    if vertical_lines:
        axes.axvline(x=300, color='firebrick', linestyle=':', label='End of first segment', linewidth = 2.5, ymin=0, ymax=0.2)
        axes.axhline(y=300, color='firebrick', linestyle=':',  label='!st Segment',xmin=0,xmax=1/3.0+0.015,linewidth=2.5)
        axes.text(-30,350,'1st Segment',rotation=360, color='firebrick',fontsize=16)

        axes.axvline(x=600, color='firebrick', linestyle='-.', label='End of second segment', linewidth = 2.5, ymin=0, ymax=0.2)
        axes.axhline(y=300, color='firebrick', linestyle='-.',  label='3rd Segment',xmin=2/3,xmax=1,linewidth=2.5)
        axes.text(620,350,'3rd Segment',rotation=360, color='firebrick',fontsize=16)

    #ax.set_ylim(0, 120.0)
    #ax.set_xlim(0, float(x_max_lim))

def plot_e2e_types(dataframe, axes):
    #ax = sns.stripplot(data=dataframe, x="Frame ID", y="level", order=level_order, jitter=False, ax=axes)
    dataframe["Seconds"]=dataframe["Frame ID"].apply(divide)
    #print(dataframe)
    #df= dataframe.value_counts(["Seconds", "End Node"])
    df= dataframe.groupby(["Seconds", "End Node"]).sum()
    yaxis_name = "Count \n per 5 secs bucket"
    df[yaxis_name] = df["Count"]
    #print(df)
    hue_order = ["Shedder", "Filter", "Detection", "Detection Filter", "Sink"] 
    ax = sns.lineplot(data=df, x="Seconds", hue="End Node",style="End Node", y=yaxis_name, ax=axes, hue_order=hue_order, linewidth = 2.5)
    #ax = sns.stripplot(data=df, x="Seconds", y="End Node", ax=axes)
    #ax.set_ylim(0, 120.0)
    #ax.set_xlim(0, float(x_max_lim))

def plot_e2e(dataframe, threshold, vertical_lines): 
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(12,14)) 
    #figure.suptitle('Big title for plot')    
    plot_e2e_latency(dataframe, axes[0], threshold, vertical_lines)
    plot_e2e_types(dataframe, axes[1])
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.xlabel("Seconds", size=34)
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

def get_others(frame_id, level):
    complete_set = set([level_column[Levels.SHEDDER],level_column[Levels.FILTER],level_column[Levels.DETECTION],level_column[Levels.DETECTION_FILTER], level_column[Levels.SINK]])
    level_set = set([level]) 
    difference = complete_set.difference(level_set)
    response = []
    for value in difference:
        response.append([frame_id, 0, value, None, None, None, 0])
    return response 

def get_dataframe(csv_file, video, add_others=True, ignore_start=True):
    with open(csv_file) as csvFile:
        reader = csv.DictReader(csvFile)
        # The headers are as follow
        data = []
        for row in reader:
            #print(row)
            if "video" not in row or video == row['video']:
                latency, level, threshold, utility  = getUsefulValues(row)
                #frame_id = int(row["frame_id"])
                frame_id = int(row["video_frame_id"])
                if (frame_id > 90) or not ignore_start:
                    data.append([frame_id,latency, level, threshold, utility, row["shed_decision"], 1])
                    if add_others:
                        data.extend(get_others(frame_id, level))
        return DataFrame(data, columns=["Frame ID","Total Latency [ms]", "End Node", "Threshold", "Utility","Shed Decision", "Count"])

def read_ground_truth(ground_truth_filename):
    periods = []
    with open(ground_truth_filename) as csvFile:
        reader = csv.DictReader(csvFile,delimiter='\t')
        for row in reader:
            objectId = row["obj_id"] 
            firstFrame = int(row["1st_frame_idx"])
            lastFrame = int(row["last_frame_idx"])
            periods.append([firstFrame, lastFrame])
    return periods

def gen_csvfile(dataframe, num_videos):
    with open('shedding_decisions.csv', 'w', newline='') as csvfile:
        fieldnames = ['video_name', 'frame_id', 'shed_decision']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in dataframe.iterrows():
            frame_id = int(row["Frame ID"])
            video_idx  = frame_id % num_videos
            frame_video_idx = int(frame_id / num_videos)
            level = row["End Node"]
            if level != level_column[Levels.SHEDDER]:
                writer.writerow({'video_name': getVideoName(video_idx) , 'frame_id': frame_video_idx, 'shed_decision': False})
            else:
                # False
                writer.writerow({'video_name': getVideoName(video_idx) , 'frame_id': frame_video_idx, 'shed_decision': True})


def main(csv_file, video, threshold, vertical_lines, gen_csv,num_videos):
    palette1 = sns.color_palette("colorblind", 8)
    sns.set_palette(palette1)
    sns.set(font_scale = 2.5)
    sns.set_style("whitegrid")
    dataframe = get_dataframe(csv_file, video, ignore_start=True)
    plot_e2e(dataframe, threshold, vertical_lines)
    if gen_csv:
        if num_videos < 1:
            raise Exception("Invalid configuration num videos is smaller than 1")
        new_dataframe = get_dataframe(csv_file, video, False, ignore_start=False)
        gen_csvfile(new_dataframe, num_videos)
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-F", dest = "csv_file", type=str, help="File that contains the csv file with the results of a run",required=True)
    parser.add_argument("-V", dest = "video", type=str, help="Name of video",required=True)
    parser.add_argument('-T', dest="threshold",type=int, default=2200)
    parser.add_argument('-E', dest = "vertical", action="store_true", default=False)
    parser.add_argument('-C', dest = "gen_csv", action="store_true", default=False)
    parser.add_argument('-N', dest = "num_videos", type=int, default=1)
    args = parser.parse_args()
    main(args.csv_file, args.video,args.threshold, args.vertical, args.gen_csv, args.num_videos)
