#!/bin/bash

frame_dirs=""
bin_files=""

S=8

for dir in seed10484-2-train-or-red seed10484-3-train-or-red seed123456789-0-train-or-red seed123456789-1-train-or-red seed123456789-2-train-or-red seed123456789-3-train-or-red seed335500-0-train-or-red seed335500-1-train-or-red seed335500-2-train-or-red seed335500-3-train-or-red seed949563-0-train-or-red seed949563-1-train-or-red seed949563-2-train-or-red seed949563-3-train-or-red
do
    bin_file=~/LoadShedderInterface/data/15_min_long_videos/$dir.bin
    frames=./frames/$dir

    frame_dirs="${frame_dirs} $frames"
    bin_files="${bin_files} $bin_file"
done

python3 plot_sv_weight_heatmap_cross_video.py -I $frame_dirs -B $bin_files -O 15_min_plots_crossvideo -S $S -T 0.8
    
