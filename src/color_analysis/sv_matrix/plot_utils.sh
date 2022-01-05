if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters. You NEED to provide the training conf yaml."
    exit
fi

TRAINING_CONF=$1
VIDEO_DIR=$(cat $TRAINING_CONF | grep training_dir | awk -F ":" '{print $2}')

util_files=$(find $VIDEO_DIR -name "frame_utils.csv")

python3 plot_frame_utils.py -U $util_files -O ./plots/yellow_only_fixed_labels
