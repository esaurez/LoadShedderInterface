#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters. You NEED to provide the training conf directory."
    exit
fi

TRAINING_CONF=$1/conf.yaml
DATA=$(cat $TRAINING_CONF | grep training_dir | awk -F ":" '{print $2}')
curr_dir=$(pwd)

for v in $(ls $DATA/)
do
    cd $DATA/$v
    mkdir frames/

    vid=$(find . -name "*.avi")
    vid=$(realpath $vid)

    python3 $curr_dir/extract_per_frame_hsv.py -V $vid -O frames/

    tar -czf frames.tar.gz frames/
    rm -r frames/
    
    cd -

done

