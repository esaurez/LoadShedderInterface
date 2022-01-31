#!/bin/bash

source ~/venv/bin/activate

data_dir=~/LoadShedderInterface/data/10fps/red_cheat
new_dir=./extended_dump/red_10fps_cheat

for i in $(ls $data_dir)
do

    curr_bin=$(ls $data_dir/$i/*.bin)
    echo $curr_bin
    python3 gen_fixed_labels_csv.py -I $new_dir/$i/new.txt -O $new_dir/$i/

    python3 ../color_analysis/fix_bin_file_labels.py -L $new_dir/$i/updated_frames_labels.csv -B $curr_bin -O $data_dir/$i
    
    python3 gen_unique_objs_per_frame.py -I $new_dir/$i/new.txt -O $data_dir/$i

    mv $data_dir/$i/updated_labels.bin $curr_bin
done

