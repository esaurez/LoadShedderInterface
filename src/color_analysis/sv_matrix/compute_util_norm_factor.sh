if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters. You NEED to provide the training conf direcotry."
    exit
fi

source ~/venv/bin/activate

TRAINING_CONF=$1

python3 compute_util_norm_factor.py -C $TRAINING_CONF > $TRAINING_CONF/normalization_factor
