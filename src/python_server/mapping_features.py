import glob

from matplotlib.pyplot import hist
import capnp
import os 
import sys
import capnp

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../"))

from capnp_serial import mapping_capnp

#mapping_capnp = capnp.load('capnp_serial/mapping.capnp')

# HSB
colors = [10,20,30,50,70,90,110,130,150,170,180,255] # red has 0-10 and 170-180 !! 

def map_hsb_histogram(histogram, absolute_pixel_count=False):
    color_counts = [0] * len(colors)# equivalent to simplefeatures below
    color = 0
    total_counts = histogram.totalCountedPixels

    h_counts = histogram.counts[0].count # assumption: counts is list with three lists: one for H, one for S, one for B.

    # Count how many pixels appear for each color in color_counts
    for i in range(len(h_counts)): # should be 256
        color_counts[color] += h_counts[i]
        if i == colors[color]: #  <= semantics for hue count.
            color+=1

    # add up red bins which are the first and the second! last
    color_counts[0] = color_counts[0] + color_counts[-2] 
    last_color = color_counts[-1]
    color_counts = color_counts[:-2]
    color_counts.append(last_color)

    if absolute_pixel_count:
        normalized_color_counts = color_counts
    else:
        normalized_color_counts = [(x / total_counts)*100.0 for x in color_counts] # prozente in vollen 100ern.

    color_counts = normalized_color_counts

    # try with two colors, only: color of interest and "other"

    sum_red=color_counts[0]
    sum_other=sum(color_counts[1:])
    color_counts_coi = [sum_red,sum_other] # red only.

    # here return either all colors or coi.

    return color_counts # all colors binned.
    #return color_counts_coi # active for color of interest.

def map_histogram(histogram, fullHistogramBinSize):
    numberOfBins = (int(256 / fullHistogramBinSize))
    simpleFeatures = [0] * numberOfBins
    for counts in histogram.counts:
        bin = 0
       # print(histogram.counts)
        for count in counts.count:
            index = int(bin / fullHistogramBinSize)
            simpleFeatures[index] += count
            bin += 1

    return simpleFeatures

# HSB
def map_full_hsb_histogram(histogram, absolute_pixel_count=False):
    return map_hsb_histogram(histogram.histo, absolute_pixel_count)


def map_full_histogram(fullHistogram, fullHistogramBinSize):
    return map_histogram(fullHistogram.histo, fullHistogramBinSize)


def map_contour(contour):
    simpleFeatures = []
    for point in contour.points:
        simpleFeatures.append(point.x)
        simpleFeatures.append(point.y)

    return simpleFeatures


def map_feature(feature, absolute_pixel_count=False, fullHistogramBinSize=None): 
   # if feature.type == mapping_capnp.Feature.Type.fullHSBHistogram:
   #     return map_full_hsb_histogram(feature.feat.hsbHistogram)

    if feature.type == mapping_capnp.Feature.Type.fullColorHistogram:
        return map_full_hsb_histogram(feature.feat.wholeHisto, absolute_pixel_count)
      #  return map_full_histogram(feature.feat.wholeHisto, fullHistogramBinSize)

    elif feature.type == mapping_capnp.Feature.Type.contours:
        return map_contour(feature.feat.contours)

    # TODO: add code for the remaining features
    raise Exception("Invalid feature type: not supported yet")


def map_features(features, absolute_pixel_count=False, fullHistogramBinSize=None):
    simple_features = []
    for feature in features.feats:
        tempFeatures = map_feature(feature, absolute_pixel_count, fullHistogramBinSize)
        simple_features.extend(tempFeatures)

    return simple_features

import csv
def map_labeled_data(labeledData, absolute_pixel_count=False, fullHistogramBinSize=None):
    observation = map_features(labeledData.feats, absolute_pixel_count, fullHistogramBinSize) # histogram
    observation.append(int(labeledData.label)) # respective label

    # todo if I want to store feature list to .csv 
  #  with open("colors_testdata.csv", "a") as fp:
  #      wr = csv.writer(fp, dialect='excel')
  #      wr.writerow(observation)

    return observation


def map_video_features(videoFeatures, absolute_pixel_count=False, fullHistogramBinSize=None): # one video at a time
    observations = []
    minSize = sys.maxsize
    for labeledData in videoFeatures.data: # data is list of labeledData - frames?
        observation = map_labeled_data(labeledData, absolute_pixel_count, fullHistogramBinSize)
        observations.append(observation)
        detections=labeledData.detections
        smallest_detection=detections.minSize
        if smallest_detection > 0:
            minSize = min(minSize,smallest_detection)

    return observations, minSize


def map_training(trainingData, absolute_pixel_count=False, fullHistogramBinSize=None):
    observations = []
    minSizes = []
    for videoFeatures in trainingData.data: # ist nur ein video enthalten!
        videoObservation, minSize = map_video_features(videoFeatures, absolute_pixel_count, fullHistogramBinSize)
        observations.extend(videoObservation)
        minSizes.append(minSize)

    return observations, minSizes

def read_samples(video_file):
    observations = []
    with open(video_file, 'rb') as f:
        trainingData = mapping_capnp.Training.read(f) # list of "Training" objects.
        for videoFeatures in trainingData.data: # ist nur ein video enthalten!
            for labeled_data in videoFeatures.data:
                observations.append(labeled_data)
    return observations

def read_training_file(videofile, absolute_pixel_count=False, fullHistogramBinSize=None):
    with open(videofile, 'rb') as f:
        trainingData = mapping_capnp.Training.read(f) # list of "Training" objects.
        fileObservations, fileMinSizes = map_training(trainingData, absolute_pixel_count, fullHistogramBinSize)
        return fileObservations

def load_training_data(path, absolute_pixel_count=False, fullHistogramBinSize=None):
    observations = []
    
    videofiles = glob.glob(path + "/*.bin")

    for videofile in videofiles:
            fileObservations = read_training_file(videofile, absolute_pixel_count, fullHistogramBinSize)
            observations.extend(fileObservations)

    return observations


def init(properties_file):
    test_load_training_data()
    pass

def test_load_training_data():
    load_training_data("training_data")

test_load_training_data()
