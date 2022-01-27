if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters. You NEED to provide the training conf directory."
    exit
fi

source ~/venv/bin/activate

TRAINING_CONF=$1/conf.yaml
VIDEO_DIR=$(cat $TRAINING_CONF | grep training_dir | awk -F ":" '{print $2}')
NUM_BINS=$(python3 ../parse_yaml.py $TRAINING_CONF num_bins)

for vid in $(ls $VIDEO_DIR)
do
    vid_dir=$VIDEO_DIR/$vid
    echo $vid

    bin_file=$(find $vid_dir -name "*.bin")

    cmd="python3 extract_sv_matrix_from_bin.py -C $TRAINING_CONF -B $bin_file -O $vid_dir --bins $NUM_BINS"
    $cmd

done

