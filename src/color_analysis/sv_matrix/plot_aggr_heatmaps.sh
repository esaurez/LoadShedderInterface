#!/bin/bash

VIDEO_DIR=/home/surveillance/LoadShedderInterface/data/seed_videos

for color in red blue
do
    for bins in 8 #4 16
    do
        neg=$(find $VIDEO_DIR -name "sv_matrix_label_False_BINS_${bins}_C_${color}.txt")
        pos=$(find $VIDEO_DIR -name "sv_matrix_label_True_BINS_${bins}_C_${color}.txt")
        python3 aggregate_sv_matrix.py -C $color -B $bins -P $pos -N $neg
    done
done
