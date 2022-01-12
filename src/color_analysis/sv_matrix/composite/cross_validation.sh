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
    NUM_BINS=$(python3 ../parse_yaml.py $TRAINING_CONF num_bins)
    color=$(python3 ../parse_yaml.py $TRAINING_CONF color_names)
    
    MATS=$4/$color
    
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
            mkdir $MATS/$vid
            cmd="python3 ../extract_sv_matrix.py -F $frames -C $TRAINING_CONF -B $bin_file -O $MATS/$vid --bins $NUM_BINS --training-split 1.0"
            $cmd
        
            python3 ../compute_per_frame_sv_mats.py -F $vid_dir/frames -C $TRAINING_CONF -O $MATS/$vid
        
            rm -r $vid_dir/frames
        done
    fi
done

### Step 2: Cross-validation training-testing
python3 cross_validation_train_test.py -C $1 $2 -M $4 --composite-or

