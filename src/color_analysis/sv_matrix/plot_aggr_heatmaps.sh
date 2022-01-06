#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters. You NEED to provide the training conf yaml."
    exit
fi

source ~/venv/bin/activate

TRAINING_CONF=$1
VIDEO_DIR=$(cat $TRAINING_CONF | grep training_dir | awk -F ":" '{print $2}')

for color in yellow #red blue # TODO Read this from the training conf directly
do
    for bins in 8 #4 16
    do
        neg=$(find $VIDEO_DIR -name "sv_matrix_label_False_BINS_${bins}_C_${color}.txt")
        pos=$(find $VIDEO_DIR -name "sv_matrix_label_True_BINS_${bins}_C_${color}.txt")
        python3 aggregate_sv_matrix.py -C $color -B $bins -P $pos -N $neg
    done
done
