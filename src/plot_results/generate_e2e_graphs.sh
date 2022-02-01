#!/usr/bin/env bash

GRAPHS_LOCATION="/tmp/e2e_graphs/"
SYNTHETIC_FILE="/mnt/fast-data/thad/videos_csv/synthetic_videos/fourth.csv"
REAL_FILE="/mnt/fast-data/thad/videos_csv/interleaved_videos/interleave-5-small-train-or-red-hsv_model-10-2200.csv"
REAL_VIDEO_NAME="interleave-5-small-train-or-red-hsv_model-10-2200"
NUM_VIDEOS="5"

mkdir -p ${GRAPHS_LOCATION}
echo "Generating synthetic graphs"
python3 ./compute_e2e_graphs_from_dataframe.py -F ${SYNTHETIC_FILE} -E
mv e2e_latency_location.pdf ${GRAPHS_LOCATION}/synthetic_latency.pdf

python3 compute_e2e_graphs.py -F ${REAL_FILE} -V ${REAL_VIDEO_NAME} -N ${NUM_VIDEOS} -C
mv e2e_latency_location.pdf ${GRAPHS_LOCATION}/realistic_latency.pdf
