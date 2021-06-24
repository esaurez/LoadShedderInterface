import glob
import capnp
import os 
import sys
import capnp

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../"))

from capnp_serial import mapping_capnp

#mapping_capnp = capnp.load('capnp_serial/mapping.capnp')


def map_histogram(histogram, fullHistogramBinSize):
    numberOfBins = (int(256 / fullHistogramBinSize))
    simpleFeatures = [0] * numberOfBins
    for counts in histogram.counts:
        bin = 0
        for count in counts.count:
            index = int(bin / fullHistogramBinSize)
            simpleFeatures[index] += count
            bin += 1

    return simpleFeatures


def map_full_histogram(fullHistogram, fullHistogramBinSize):
    return map_histogram(fullHistogram.histo, fullHistogramBinSize)


def map_contour(contour):
    simpleFeatures = []
    for point in contour.points:
        simpleFeatures.append(point.x)
        simpleFeatures.append(point.y)

    return simpleFeatures


def map_feature(feature, fullHistogramBinSize=None):
    if feature.type == mapping_capnp.Feature.Type.fullColorHistogram:
        return map_full_histogram(feature.feat.wholeHisto, fullHistogramBinSize)

    if feature.type == mapping_capnp.Feature.Type.contours:
        return map_contour(feature.feat.contours)

    # TODO: add code for the remaining features


def map_features(features, fullHistogramBinSize=None):
    simple_features = []
    for feature in features.feats:
        tempFeatures = map_feature(feature, fullHistogramBinSize)
        simple_features.extend(tempFeatures)

    return simple_features


def map_labeled_data(labeledData, fullHistogramBinSize=None):
    observation = map_features(labeledData.feats, fullHistogramBinSize)
    observation.append(int(labeledData.label))

    return observation


def map_video_features(videoFeatures, fullHistogramBinSize=None):
    observations = []
    for labeledData in videoFeatures.data:
        observation = map_labeled_data(labeledData, fullHistogramBinSize)
        observations.append(observation)

    return observations


def map_training(trainingData, fullHistogramBinSize=None):
    observations = []
    for videoFeatures in trainingData.data:
        observations.extend(map_video_features(videoFeatures, fullHistogramBinSize))

    return observations


def load_training_data(path, fullHistogramBinSize=None):
    observations = []
    videofiles = glob.glob(path + "/*.bin")
    for videofile in videofiles:
        with open(videofile, 'rb') as f:
            trainingData = mapping_capnp.Training.read(f)
            observations.extend(map_training(trainingData, fullHistogramBinSize))

    return observations


def init(properties_file):
    pass
