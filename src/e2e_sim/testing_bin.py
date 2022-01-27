import argparse
from os import listdir
from os.path import join, isfile, isdir
import yaml
import pandas as pd
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../"))
import model.build_model
import python_server.mapping_features

def main(bin_file):
    observations = python_server.mapping_features.read_samples(bin_file)
    print (observations[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-B", dest="bin_file", help="Path to bin file", required=True)
    args = parser.parse_args()
    main(args.bin_file)
