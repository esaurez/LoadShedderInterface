#!/bin/bash

curr_dir=$(pwd)
#DATA=~/LoadShedderInterface/data/seed_videos/
DATA=~/LoadShedderInterface/data/red_and_blue

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

