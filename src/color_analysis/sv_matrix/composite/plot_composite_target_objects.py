import argparse
import os, sys
from os import listdir
from os.path import join, isfile, isdir
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../../../"))
import model.build_model
import python_server.mapping_features

def get_bin_file(d):
    return [join(d, f) for f in listdir(d) if isfile(join(d, f)) and f.endswith(".bin")][0]

def main(video_dirs, output_dir, composite_or):
    labels = []
    for vid_dir in video_dirs:
        if not isdir(vid_dir):
            return
        labels.append([])
        bin_file = get_bin_file(vid_dir)
        observations = python_server.mapping_features.read_samples(bin_file)
        for o in observations:
            labels[-1].append(o.label)

    if composite_or:
        composite_labels = [False for i in range(len(labels[0]))]
        for idx in range(len(labels[0])):
            for f in range(len(labels)):
                if labels[f][idx]:
                    composite_labels[idx] = True
    else:
       composite_labels = [True for i in range(len(labels[0]))]
       for idx in range(len(labels[0])):
            for f in range(len(labels)):
                if not labels[f][idx]:
                    composite_labels[idx] = False

    print (len([x for x in composite_labels if x]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting the occcurrence of composite objects")
    parser.add_argument("-V", "--video-dirs", dest="video_dirs", help="Paths to data for each video", nargs="+")
    parser.add_argument("-O", "--out-dir", dest="output_dir", help="Path to output directory")
    parser.add_argument("--composite-or", dest="composite_or", action="store_true", default=False)
    args = parser.parse_args()
    main(args.video_dirs, args.output_dir, args.composite_or)
