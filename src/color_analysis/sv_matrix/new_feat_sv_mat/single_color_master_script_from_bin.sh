#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters. You NEED to provide the training conf directory."
    exit
fi

source ~/venv/bin/activate
TRAINING_CONF=$1

# Step 1 : Extract per-video SV matrix
./compute_per_video_sv_mats.sh $TRAINING_CONF

# Step 2: Aggregate the SV Matrix of videos to build the model
../plot_aggr_heatmaps.sh $TRAINING_CONF

# Step 3: Computing utility of test frames
./compute_test_frames_util.sh $TRAINING_CONF

# Step 4 : Compute norm factor
../compute_util_norm_factor.sh $TRAINING_CONF

# Step 5: Compute utils cdf
python3 ../compute_utils_cdf.py -C $TRAINING_CONF/conf.yaml -O /tmp/
