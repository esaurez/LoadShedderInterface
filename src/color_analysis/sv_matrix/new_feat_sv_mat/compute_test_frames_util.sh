if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters. You NEED to provide the training conf directory."
    exit
fi

source ~/venv/bin/activate

TRAINING_CONF=$1/conf.yaml
VIDEO_DIR=$(cat $TRAINING_CONF | grep training_dir | awk -F ":" '{print $2}')
NUM_BINS=$(python3 ../parse_yaml.py $TRAINING_CONF num_bins)
COLORS=$(python3 ../parse_yaml.py $TRAINING_CONF color_names)

util_files=""
for color in $COLORS
do
    util_filename=$(realpath $1/utils_${color}_BINS_${NUM_BINS}.txt)
    util_files="$util_files $util_filename"
done

for vid in $(ls $VIDEO_DIR)
do
    vid_dir=$VIDEO_DIR/$vid

    bin_file=$(find $vid_dir -name "*.bin")
    frames=$vid_dir/frames

    cmd="python3 compute_test_frames_util.py -C $TRAINING_CONF -B $bin_file -O $vid_dir --bins $NUM_BINS --util-files $util_files -V $vid -A"
    $cmd
done

