import argparse
import yaml
from os.path import join, isfile, isdir
from os import listdir
import pandas as pd

def main(training_conf_dir):
    with open(join(training_conf_dir, "conf.yaml")) as f:
        training_conf = yaml.safe_load(f)

    training_dir = training_conf["training_dir"]
    training_split = training_conf["training_split"]
    
    vids = [join(training_dir, d) for d in listdir(training_dir) if isdir(join(training_dir, d))]

    max_util = None
    for vid in vids:
        utils_file = join(vid, "frame_utils.csv")
        df = pd.read_csv(utils_file)
        num_training_frames = int(len(df)*training_split)
        df = df[df["frame_id"] < num_training_frames]
        vidmax = df["utility"].max()

        if max_util == None or vidmax > max_util:
            max_util = vidmax

    print (max_util)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", dest="training_conf", required=True, help="Path to the configuration folder")
    args = parser.parse_args()

    main(args.training_conf)
