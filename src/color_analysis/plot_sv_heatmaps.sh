#!/bin/bash

for dir in seed10484-2-train-or-red seed10484-3-train-or-red seed123456789-0-train-or-red seed123456789-1-train-or-red seed123456789-2-train-or-red seed123456789-3-train-or-red seed335500-0-train-or-red seed335500-1-train-or-red seed335500-2-train-or-red seed335500-3-train-or-red seed949563-0-train-or-red seed949563-1-train-or-red seed949563-2-train-or-red seed949563-3-train-or-red; do

    bin_file=~/LoadShedderInterface/data/15_min_long_videos/$dir.bin
    frames=./frames/$dir

    for S in 8
    do
        python3 plot_sv_weight_heatmap.py -I $frames -B $bin_file -O 15_min_plots -S $S -C red -F 10 --max-weight 0.36
    done
    
done
