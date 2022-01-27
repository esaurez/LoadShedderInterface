#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters. You NEED to provide the training conf directory."
    exit
fi

source ~/venv/bin/activate
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
TRAINING_CONF=$1/conf.yaml
VIDEO_DIR=$(cat $TRAINING_CONF | grep training_dir | awk -F ":" '{print $2}')
NUM_BINS=$(python3 $SCRIPT_DIR/parse_yaml.py $TRAINING_CONF num_bins)
COLORS=$(python3 $SCRIPT_DIR/parse_yaml.py $TRAINING_CONF color_names)

for color in $COLORS 
do
        neg=$(find $VIDEO_DIR -name "sv_matrix_label_False_BINS_${NUM_BINS}_C_${color}.txt")
        pos=$(find $VIDEO_DIR -name "sv_matrix_label_True_BINS_${NUM_BINS}_C_${color}.txt")
        python3 $SCRIPT_DIR/aggregate_sv_matrix.py -C $color -B $NUM_BINS -P $pos -N $neg -O $1/
done
