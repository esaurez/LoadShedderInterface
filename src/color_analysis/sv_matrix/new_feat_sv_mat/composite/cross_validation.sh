#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Illegal number of parameters. You NEED to provide the two training conf directories, composition operator and the dump dir."
    exit
fi

source ~/venv/bin/activate

for TRAINING_CONF_DIR in $1 $2
do
    TRAINING_CONF=$TRAINING_CONF_DIR/conf.yaml
    VIDEO_DIR=$(cat $TRAINING_CONF | grep training_dir | awk -F ":" '{print $2}')
    NUM_BINS=$(python3 ../../parse_yaml.py $TRAINING_CONF num_bins)
    color=$(python3 ../../parse_yaml.py $TRAINING_CONF color_names)
    
    MATS=$4/$color
    
    PREPROCESS=0

    if [[ $PREPROCESS = 1 ]]
    then
        rm -r $MATS
        mkdir -p $MATS
        
        pids=""
    
        # Step 1 : Computer per video SV matrix
        for vid in $(ls $VIDEO_DIR)
        do
            mkdir $MATS/$vid
            echo $vid
            vid_dir=$VIDEO_DIR/$vid/
            bin_file=$(find $vid_dir -name "*.bin")
            echo $bin_file
            cmd="python3 ../extract_sv_matrix_from_bin.py -C $TRAINING_CONF -B $bin_file -O $MATS/$vid --bins $NUM_BINS --training-split 1.0"
            $cmd &
            pids="$pids $!"
        
        done
    
        for pid in $pids
        do
            wait $pid
        done
    fi

done
# Step 2: Cross-validation training-testing
python3 cross_validation_train_test_composite.py -C $1 $2 -M $4 --composite-or

