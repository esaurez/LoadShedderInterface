import sys
from configobj import ConfigObj

import capnp

import model.build_model
import mapping_features

mapping_capnp = capnp.load('capnp_serial/mapping.capnp')

"""
train the model
"""


def train(properties_file):
    config = ConfigObj(properties_file)
    fullHistogramBinSize = int(config["fullHistogramBinSize"])
    featureCorrespondingBinSize = list(map(int, config["featureCorrespondingBinSize"]))
    utilityNormalizationUpperBound = int(config["utilityNormalizationUpperBound"])
    trainingDataPath = config["trainingDataPath"]
    generatedModelPath = config["generatedModelPath"]

    # load stored label data
    observations = mapping_features.load_training_data(trainingDataPath, fullHistogramBinSize)

    # train the model

    model.build_model.train(observations, featureCorrespondingBinSize, generatedModelPath,
                            utilityNormalizationUpperBound)

if __name__ == "__main__":
    properties_file = sys.argv[1]
    train(properties_file)
