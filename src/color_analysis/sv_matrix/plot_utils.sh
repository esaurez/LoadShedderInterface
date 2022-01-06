if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters. You NEED to provide the training conf yaml."
    exit
fi

source ~/venv/bin/activate

TRAINING_CONF=$1
VIDEO_DIR=$(cat $TRAINING_CONF | grep training_dir | awk -F ":" '{print $2}')
TRAINING_SPLIT=$(cat $TRAINING_CONF | grep training_split | awk -F ":" '{print $2}')

plot_subdir=$(basename $TRAINING_CONF | awk -F "." '{print $1}')
mkdir -p plots/$plot_subdir

util_files=$(find $VIDEO_DIR -name "frame_utils.csv")

python3 plot_frame_utils.py -U $util_files -O ./plots/$plot_subdir -S $TRAINING_SPLIT
