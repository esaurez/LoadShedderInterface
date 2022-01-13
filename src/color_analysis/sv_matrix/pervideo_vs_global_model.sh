#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters. You NEED to provide the training conf directory and dump dir."
    exit
fi

source ~/venv/bin/activate

TRAINING_CONF=$1/conf.yaml
VIDEO_DIR=$(cat $TRAINING_CONF | grep training_dir | awk -F ":" '{print $2}')
NUM_BINS=$(python3 parse_yaml.py $TRAINING_CONF num_bins)
TRAINING_SPLIT=$(python3 parse_yaml.py $TRAINING_CONF training_split)

MATS=$2

PREPROCESS=0

if [[ $PREPROCESS = 1 ]]
then
    rm -r $MATS
    mkdir -p $MATS
    
    # Step 1 : Computer per video SV matrix
    for vid in $(ls $VIDEO_DIR)
    do
        echo $vid
        vid_dir=$VIDEO_DIR/$vid
        cd $vid_dir
        tar -xf frames.tar.gz
        cd -
        bin_file=$(find $vid_dir -name "*.bin")
        frames=$vid_dir/frames
        cmd="python3 extract_sv_matrix.py -F $frames -C $TRAINING_CONF -B $bin_file -O $vid_dir --bins $NUM_BINS --training-split $TRAINING_SPLIT"
        $cmd
    
        mkdir $MATS/$vid
        python3 ./compute_per_frame_sv_mats.py -F $vid_dir/frames -C $TRAINING_CONF -O $MATS/$vid
    
        rm -r $vid_dir/frames
    done
fi

# Step 2: Cross-validation training-testing
python3 pervideo_vs_global_model.py -C $1 -M $MATS

