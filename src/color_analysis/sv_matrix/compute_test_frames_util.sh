VIDEO_DIR=~/LoadShedderInterface/data/seed_videos

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
        cmd="python3 compute_test_frames_util.py -F $frames -C ../training_confs/red_only.yaml -B $bin_file -O $vid_dir --bins $bins --util-files utils_red_BINS_${bins}.txt utils_blue_BINS_${bins}.txt -V $vid"
        $cmd
    done

    rm -r $vid_dir/frames

    #break
done

