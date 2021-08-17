import os
import sys
from configobj import ConfigObj

import capnp

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../"))
import model.build_model
import mapping_features

from capnp_serial import mapping_capnp


"""
train the model
"""


def train(properties_file):
    config = ConfigObj(properties_file)


    print(config) 

   # fullHistogramBinSize = int(config["fullHistogramBinSize"])
   # featureCorrespondingBinSize = list(map(int, config["featureCorrespondingBinSize"]))
    utilityNormalizationUpperBound = int(config["utilityNormalizationUpperBound"])
    trainingDataPath = config["trainingDataPath"]
    generatedModelPath = config["generatedModelPath"]
    split_values = config["splitvalues"]
    split_values = [int(x) for x in split_values]

    # load stored label data
    observations, min_sizes = mapping_features.load_training_data(trainingDataPath) #, fullHistogramBinSize)

    smallest_size = min(min_sizes)
    biggest_minsize = max(min_sizes)
    # train the model
    
  #  split_values = [smallest_size,biggest_minsize]
 #   split_values = [smallest_size] # smallest only.
 #   split_values = min_sizes # take all of them
 #   split_values.sort()
 #   split_values = split_values[0:3]
 #   split_values = [sum(min_sizes) // len(min_sizes)] # try average
  #  split_values=[biggest_minsize]
   # split_values = min_sizes
   # split_values.sort()
   # split_values = 

  #  bins_per_color = 100
  ##  step = max(min_sizes)
  #  step = sum(min_sizes) // len(min_sizes) # tried min, max, and mean.
  #  step = 1000
  #  split_values = [i * step for i in range(1,bins_per_color)]

  #model.build_model.train(observations, featureCorrespondingBinSize, generatedModelPath,
    #                        utilityNormalizationUpperBound,split_values=split_values)
    
    model.build_model.train(observations, generatedModelPath,
                            utilityNormalizationUpperBound,split_values=split_values)

   # split_values_to_string = ""
   # for split_value in split_values:
   #     split_values_to_string = split_values_to_string + str(split_value) + ","
    #print(split_values_to_string)
    #f = open(properties_file,'a')
    #f.writelines(['\n',f"splitvalues = {split_values_to_string}"])
    #f.close()



if __name__ == "__main__":
    properties_file = sys.argv[1]
    train(properties_file)
