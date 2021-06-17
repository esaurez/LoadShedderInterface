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


"""
get the utility threshold for the given drop ratio
"""


def get_utility_threshold(dropRatio):
    return model.build_model.get_utility_threshold(dropRatio)


"""
predict the utility given a list of complex features (e.g., histogram, contour, ...)
"""


def get_utility(features):
    featureList = mapping_features.map_features(features)
    return model.build_model.get_utility(featureList)


"""
load the predicted utilities and the computed utility thresholds
this function must be called before performing load shedding
"""


def init_shedding(properties_file):
    config = ConfigObj(properties_file)
    featureCorrespondingBinSize = list(map(int, config["featureCorrespondingBinSize"]))
    generatedModelPath = config["generatedModelPath"]

    model.build_model.init_shedding(generatedModelPath, featureCorrespondingBinSize)


if __name__ == "__main__":
    properties_file = sys.argv[1]
    train(properties_file)
