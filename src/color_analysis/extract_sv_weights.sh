#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters. Exiting."
    exit
fi

bin_dir=$1

if [ -d "$bin_dir" ] 
then
    echo "Directory $bin_dir exists."
else
    echo "Provided directory $bin_dir does not exist. Exiting."
    exit
fi

for avi_file in $bin_dir/*.avi
do
    name=$(basename $avi_file)
    prefix=$(echo $name | awk -F "_" '{print $1}')
    bin_file="$bin_dir/${prefix}.bin"

    rm -r frames/$prefix
    mkdir -p frames/$prefix

    python3 extract_sv_weights.py -V $avi_file -B $bin_file -O frames/$prefix

done
