import os
from os.path import join
import sys
import capnp

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../"))

from capnp_serial import mapping_capnp

import argparse
import pandas as pd

def main(bin_file, labels, outdir):
    labels_df = pd.read_csv(labels)
    frame_to_label = {}
    for idx, row in labels_df.iterrows():
        if row["label"] == 0:
            label = False
        else:
            label = True
        frame_to_label[row["frame_idx"]] = label

    with open(bin_file, 'rb') as f:
        training_data = mapping_capnp.Training.read(f)
        data = training_data.to_dict()
    
    for vid_idx in range(len(data["data"])):
        for frame_idx in range(len(data["data"][vid_idx]["data"])):
            if frame_idx in frame_to_label:
                data["data"][vid_idx]["data"][frame_idx]["label"] = frame_to_label[frame_idx]

    out = mapping_capnp.Training.new_message()
    out.from_dict(data)

    with open(join(outdir, "updated_labels.bin"), "w+b") as f:
        out.write(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fixing frame labels using manual correction")
    parser.add_argument("-B", "--bin-file", dest="bin_file", help="Path to bin file for the video")
    parser.add_argument("-L", "--correct-labels-csv", dest="labels_csv", help="CSV containing labels to fix")
    parser.add_argument("-O", dest="outdir", help="Output directory to store the modified bin file")
    args = parser.parse_args()
    main(args.bin_file, args.labels_csv, args.outdir)
