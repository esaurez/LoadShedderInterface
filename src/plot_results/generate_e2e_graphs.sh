#!/usr/bin/env bash

GRAPHS_LOCATION="/tmp/e2e_graphs/"
SYNTHETIC_FILE="/mnt/fast-data/thad/videos_csv/synthetic_videos/fourth.csv"
SYNTHETIC_VIDEO_NAME="new_merged-train-or-red-red_model-24-5000"
REAL_VIDEOS_DIR="/mnt/fast-data/thad/videos_csv/real_videos"

mkdir -p ${GRAPHS_LOCATION}
echo "Generating synthetic graphs"
python3 compute_e2e_graphs.py -F ${SYNTHETIC_FILE} -V ${SYNTHETIC_VIDEO_NAME} -E
mv e2e_latency_location.pdf ${GRAPHS_LOCATION}/synthetic_latency.pdf


echo "Generating real video graphs"
for i in $(ls ${REAL_VIDEOS_DIR})
do
    VIDEO_NAME="${i%.*}"
    echo "Processing [ ${VIDEO_NAME} ]"
    python3 compute_e2e_graphs.py -F ${REAL_VIDEOS_DIR}/${i} -V ${VIDEO_NAME}
    mv e2e_latency_location.pdf ${GRAPHS_LOCATION}/${VIDEO_NAME}.pdf
done
