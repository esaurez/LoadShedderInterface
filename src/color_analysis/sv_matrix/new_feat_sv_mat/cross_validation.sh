#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters. You NEED to provide the training conf directory and dump dir."
    exit
fi

source ~/venv/bin/activate

TRAINING_CONF=$1/conf.yaml
VIDEO_DIR=$(cat $TRAINING_CONF | grep training_dir | awk -F ":" '{print $2}')
NUM_BINS=$(python3 ../parse_yaml.py $TRAINING_CONF num_bins)

MATS=$2

PREPROCESS=1

if [[ $PREPROCESS = 1 ]]
then
    rm -r $MATS
    mkdir -p $MATS
    
    pids=""

    # Step 1 : Computer per video SV matrix
    for vid in $(ls $VIDEO_DIR)
    do
        echo $vid
        mkdir $MATS/$vid
        vid_dir=$VIDEO_DIR/$vid/
        bin_file=$(find $vid_dir -name "*.bin")
        echo $bin_file
        cmd="python3 extract_sv_matrix_from_bin.py -C $TRAINING_CONF -B $bin_file -O $MATS/$vid --bins $NUM_BINS --training-split 1.0"
        $cmd &
        pids="$pids $!"
    
    done

    for pid in $pids
    do
        wait $pid
    done
fi

# Step 2: Cross-validation training-testing
python3 cross_validation_train_test.py -C $1 -M $MATS

