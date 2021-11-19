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
from os.path import basename
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# TODO : Fix this definition of colors everywhere. Centralize it in a constants.py file.
# The main distribution of color values is in mapping_features file
COLORS = ['red', 'orange','yellow','spring_green','green','ocean_green','light_blue','blue','purple','magenta']

def per_cluster_util(kmeans_labels, gt_labels):
    cluster_counts = {}
    for idx in range(len(kmeans_labels)):
        cluster_id = kmeans_labels[idx]
        gt_label = gt_labels[idx]
        if cluster_id not in cluster_counts:
            cluster_counts[cluster_id] = [0, 0] # [positive, total]
        cluster_counts[cluster_id][1] += 1
        cluster_counts[cluster_id][0] += gt_label
    
    num_clusters = len(cluster_counts)
    utils = []
    for cluster in range(num_clusters):
        util = cluster_counts[cluster][0]/float(cluster_counts[cluster][1])
        utils.append(util)
    return utils

def do_kmeans(video_file, absolute_pixel_count):
    # Extracting the features from training data samples
    observations = python_server.mapping_features.read_training_file(video_file, absolute_pixel_count=absolute_pixel_count)
    frame_id=0

    points = []
    labels = []

    for observation in observations:
        # observation is in the form of [feat1, feat2, ..., featN, label]
        feature_list = observation[0:-1]
        label = observation[-1]

        points.append(feature_list)
        labels.append(label)

        #for color in COLORS:
        #    # Extracting the pixel fraction for color
        #    color_position = COLORS.index(color)
        #    pixel_fraction = feature_list[color_position]
        frame_id += 1

    scaler = StandardScaler()
    scaled_points = scaler.fit_transform(points)

    kmeans = KMeans(init="random", n_clusters=4)
    kmeans.fit(scaled_points)
    #kmeans.fit(points)
    print(per_cluster_util(kmeans.labels_, labels))
 
def main(training_data, outdir):
    video_files = glob.glob(training_data+"/*.bin")
    for video_file in video_files:
        for absolute_pixel_count in [False]:
            do_kmeans(video_file, absolute_pixel_count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzing the distribution of Pixel Fraction")
    parser.add_argument("-D", "--training-data", dest="training_data", help="Path to training data")
    parser.add_argument("-O", "--out-dir", dest="output_dir", help="Path to output data that would contain plots")
    
    args = parser.parse_args()
    main(args.training_data, args.output_dir)
