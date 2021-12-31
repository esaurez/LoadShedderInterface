if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters. You NEED to provide the training conf yaml."
    exit
fi

TRAINING_CONF=$1
VIDEO_DIR=$(cat $TRAINING_CONF | grep training_dir | awk -F ":" '{print $2}')

for vid in $(ls $VIDEO_DIR)
do
    vid_dir=$VIDEO_DIR/$vid

    cd $vid_dir
    tar -xf frames.tar.gz
    cd -

    bin_file=$(find $vid_dir -name "*.bin")
    frames=$vid_dir/frames

    for bins in 8
    do
        cmd="python3 compute_test_frames_util.py -F $frames -C $TRAINING_CONF -B $bin_file -O $vid_dir --bins $bins --util-files utils_yellow_BINS_${bins}.txt -V $vid -A"
        #cmd="python3 compute_test_frames_util.py -F $frames -C $TRAINING_CONF -B $bin_file -O $vid_dir --bins $bins --util-files utils_red_BINS_${bins}.txt utils_blue_BINS_${bins}.txt -V $vid"
        $cmd
    done

    rm -r $vid_dir/frames

    #break
done

