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


def train(properties_file, min_bin_size=0.1):
    config = ConfigObj(properties_file)


    print(config) 

   # fullHistogramBinSize = int(config["fullHistogramBinSize"])
   # featureCorrespondingBinSize = list(map(int, config["featureCorrespondingBinSize"]))
    utilityNormalizationUpperBound = int(config["utilityNormalizationUpperBound"])
    trainingDataPath = config["trainingDataPath"]
    generatedModelPath = config["generatedModelPath"]
    split_values = config["splitvalues"]
    min_bin_size = float(config["minimumBinSize"])
    split_values = [int(x) for x in split_values]

    # load stored label data
    observations = mapping_features.load_training_data(trainingDataPath) #, fullHistogramBinSize)
    # model.build_model.train(observations, featureCorrespondingBinSize, generatedModelPath,
    #                        utilityNormalizationUpperBound,split_values=split_values)
    

    # iterate here over colors.

    colors = ['red', 'orange','yellow','spring_green','green','ocean_green','light_blue','blue','purple','magenta']

    
    model.build_model.train(observations, generatedModelPath, 
                              utilityNormalizationUpperBound, split_values=split_values, min_bin_size = min_bin_size)

      # should return the split value to be written into the properties file. or add that into the model-folder! 

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
