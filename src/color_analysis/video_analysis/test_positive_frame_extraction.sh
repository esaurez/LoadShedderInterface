#!/bin/bash

dir=seed10484-3-train-or-red
bin_file=~/LoadShedderInterface/data/15_min_long_videos/$dir.bin
video_file=~/LoadShedderInterface/data/15_min_long_videos/${dir}_30_fg.avi

python3 extract_positive_frames.py -V $video_file -B $bin_file -O output
