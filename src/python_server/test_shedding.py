import os
import sys
from pathlib import Path
from configobj import ConfigObj
import argparse
import glob
import random

import capnp

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../"))
from model import build_model
import mapping_features
import csv

from capnp_serial import mapping_capnp


def shed(properties_file, drop_ratio, ls_result_path):
    config = ConfigObj(properties_file)

    trainingDataPath = config["trainingDataPath"]
    trainingDataPath = config["testDataPath"] # use test  data instead of training data.
    generatedModelPath = config["generatedModelPath"]
    mode = config["mode"]

    # initialize shedding
    build_model.init_shedding(generatedModelPath)

    # load features
    observations = mapping_features.load_training_data(trainingDataPath) 

    # get the utility threshold
    utility_threshold, th_ratio = build_model.get_utility_threshold(drop_ratio, mode) # new! this now requires the utility threshold AND the respective ratio.

    # for later use (see below)
    # ratio_share = drop_ratio / th_ratio # the drop ratio might be smaller than the real ratio. so only shed the respective share.

    # create the result directory if not exists
    Path(ls_result_path).mkdir(parents=True, exist_ok=True)

    # open a file to save shedding results
    f = open(os.path.join(ls_result_path, "shedding_results.csv"), 'w')
    csv_writer = csv.writer(f)

    # write the header
    csv_writer.writerow(['threshold', 'utility', 'shed_decision', 'useful_gt'])

    #stats
    number_of_frames= 0
    number_of_dropped_frames= 0
   
    for observation in observations:
        utility = build_model.get_utility(observation[:-1], mode) 

        ratio_share = drop_ratio / th_ratio # the drop ratio might be smaller than the real ratio. so only shed the respective share.
        if utility > utility_threshold:  # keep
            csv_writer.writerow([utility_threshold, utility, False, bool(observation[-1])])
        else: 
          # keep this option in case we always shed too much:! 
            # 2021-08-26-HR adapt for probabilistic shedding of the utility on the threshold itself to overcome the problem of 
            # imprecise shedding.
          #  if utility == utility_threshold:
          #      print("randomized")
          #      rn = random.uniform(0, 1) # maybe draw those in advance on the server.
          #      if rn > ratio_share: # keep
          #          csv_writer.writerow([utility_threshold, utility, False, bool(observation[-1])])
          #          print("keep randomly")
          #      else: # shed
          #          csv_writer.writerow([utility_threshold, utility, True, bool(observation[-1])])
          #          number_of_dropped_frames +=1
          #          print("shed randomly")
          #  else:
          
          # shed:
            csv_writer.writerow([utility_threshold, utility, True, bool(observation[-1])])
            number_of_dropped_frames +=1
        number_of_frames +=1

    f.close()
  #  print(picked_colors)
  #  print(found_classes)

    print ("The drop ratio = " + str (number_of_dropped_frames/ number_of_frames) )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--properties_file", dest="properties_file", required=True, type=str,
                        help="Path to the properties file.")
    parser.add_argument("-d", "--drop_ratio", dest="drop_ratio", required=True, type=float,
                        help=" the drop ratio.")

    parser.add_argument("-r", "--result_path", dest="ls_result_path", required=True, type=str,
                        help="Path to save the shedding resul@t.")

    args = parser.parse_args()

    shed(args.properties_file, args.drop_ratio, args.ls_result_path)
