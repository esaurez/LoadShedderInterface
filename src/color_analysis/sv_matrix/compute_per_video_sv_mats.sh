if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters. You NEED to provide the training conf yaml."
    exit
fi

source ~/venv/bin/activate

TRAINING_CONF=$1
VIDEO_DIR=$(cat $TRAINING_CONF | grep training_dir | awk -F ":" '{print $2}')

for vid in $(ls $VIDEO_DIR)
do
    vid_dir=$VIDEO_DIR/$vid
    echo $vid

    cd $vid_dir
    tar -xf frames.tar.gz
    cd -

    bin_file=$(find $vid_dir -name "*.bin")
    frames=$vid_dir/frames

    for bins in 8
    do
        cmd="python3 extract_sv_matrix.py -F $frames -C $TRAINING_CONF -B $bin_file -O $vid_dir --bins $bins"
        $cmd
    done

    cp $vid_dir/heatmap*.png ./plots/

    rm -r $vid_dir/frames

    #break
done

