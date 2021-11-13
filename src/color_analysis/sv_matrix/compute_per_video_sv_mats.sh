VIDEO_DIR=~/LoadShedderInterface/data/seed_videos

for vid in $(ls $VIDEO_DIR)
do
    vid_dir=$VIDEO_DIR/$vid
    echo $vid

    cd $vid_dir
    tar -xf frames.tar.gz
    cd -

    bin_file=$(find $vid_dir -name "*.bin")
    frames=$vid_dir/frames

    cmd="python3 extract_sv_matrix.py -F $frames -C ../training_confs/red_only.yaml -B $bin_file -O $vid_dir"
    $cmd

    cp $vid_dir/heatmap*.png ./plots/

    rm -r $vid_dir/frames

done

