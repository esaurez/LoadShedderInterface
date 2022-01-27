if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters. You NEED to provide the training conf directory and the output directory."
    exit
fi

source ~/venv/bin/activate

TRAINING_CONF=$1/conf.yaml
OUTDIR=$2
VIDEO_DIR=$(cat $TRAINING_CONF | grep training_dir | awk -F ":" '{print $2}')

for vid in $(ls $VIDEO_DIR)
do
    vid_dir=$VIDEO_DIR/$vid
    echo $vid
    mkdir -p $OUTDIR/$vid

    bin_file=$(find $vid_dir -name "*.bin")
    vid_file=$(find $vid_dir -name "*.avi")
    
    echo $bin_file
    echo $vid_file

    python3 extract_negative_frames.py -V $vid_file -B $bin_file  -O $OUTDIR/$vid
done

