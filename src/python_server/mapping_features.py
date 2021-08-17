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
colors = [10,20,30,50,70,90,110,130,150,170,180,255] # red has 0-10 and 180-255

def map_hsb_histogram(histogram):
    color_counts = [0] * len(colors)# equivalent to simplefeatures below
    color = 0
    total_counts = histogram.totalCountedPixels
    
  #  print(len(histogram.counts[0].count))
    h_counts = histogram.counts[0].count # assumption: counts is list with three lists: one for H, one for S, one for B.
    #print (h_counts)
  #  print(histogram.channels)
  #  print(histogram.ranges)
  #  print(histogram.dimensions) 
   # print(histogram.histSize) # no of total pixels

    for i in range(len(h_counts)): # should be 256
        color_counts[color] += h_counts[i]
        if i == colors[color]: #  <= semantics for hue count.
            color+=1
    
    # add up red bins which are the first and the last
    color_counts[0] = color_counts[0] + color_counts[-1]
    color_counts = color_counts[:-1]

  #  print(color_counts)
  #  print(sum(color_counts))
  #  print(total_counts)
    
    normalized_color_counts = [(x / total_counts)*100 for x in color_counts] # prozente in vollen 100ern.
  #  print(normalized_color_counts)
    color_counts = normalized_color_counts
    return color_counts # list of 13 bins with the count per bin. # maybe make red together.
    

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
def map_full_hsb_histogram(histogram):
    return map_hsb_histogram(histogram.histo)


def map_full_histogram(fullHistogram, fullHistogramBinSize):
    return map_histogram(fullHistogram.histo, fullHistogramBinSize)


def map_contour(contour):
    simpleFeatures = []
    for point in contour.points:
        simpleFeatures.append(point.x)
        simpleFeatures.append(point.y)

    return simpleFeatures


def map_feature(feature, fullHistogramBinSize=None): # TODO requires a feature type hsbhistogram with an hsbHistogram
   # if feature.type == mapping_capnp.Feature.Type.fullHSBHistogram:
   #     return map_full_hsb_histogram(feature.feat.hsbHistogram)

    if feature.type == mapping_capnp.Feature.Type.fullColorHistogram:
        return map_full_hsb_histogram(feature.feat.wholeHisto)
      #  return map_full_histogram(feature.feat.wholeHisto, fullHistogramBinSize)

    elif feature.type == mapping_capnp.Feature.Type.contours:
        return map_contour(feature.feat.contours)

    # TODO: add code for the remaining features
    raise Exception("Invalid feature type: not supported yet")


def  map_features(features, fullHistogramBinSize=None):
    simple_features = []
    for feature in features.feats:
        tempFeatures = map_feature(feature, fullHistogramBinSize)
        simple_features.extend(tempFeatures)

    return simple_features


def map_labeled_data(labeledData, fullHistogramBinSize=None):
    observation = map_features(labeledData.feats, fullHistogramBinSize)
    observation.append(int(labeledData.label))

    return observation


def map_video_features(videoFeatures, fullHistogramBinSize=None): # one video at a time
    observations = []
    minSize = sys.maxsize
    for labeledData in videoFeatures.data: # data is list of labeledData - frames?
        observation = map_labeled_data(labeledData, fullHistogramBinSize)
        observations.append(observation)
        detections=labeledData.detections
        smallest_detection=detections.minSize
        if smallest_detection > 0:
            minSize = min(minSize,smallest_detection)

    return observations, minSize


def map_training(trainingData, fullHistogramBinSize=None):
    observations = []
    minSizes = []
    for videoFeatures in trainingData.data: # ist nur ein video enthalten!
        videoObservation, minSize = map_video_features(videoFeatures, fullHistogramBinSize)
        observations.extend(videoObservation)
        minSizes.append(minSize)

    return observations, minSizes


def load_training_data(path, fullHistogramBinSize=None):
    observations = []
    
    videofiles = glob.glob(path + "/*.bin")
    min_sizes = []

    for videofile in videofiles:
        with open(videofile, 'rb') as f:
            trainingData = mapping_capnp.Training.read(f) # list of "Training" objects.
            fileObservations, fileMinSizes = map_training(trainingData, fullHistogramBinSize)
            
            observations.extend(fileObservations)
            min_sizes.extend(fileMinSizes)
    #print(f'detected min sizes: {min_sizes}')
           # observations.extend(map_training(trainingData, fullHistogramBinSize))
            

    return observations, min_sizes


def init(properties_file):
    test_load_training_data()
    pass

def test_load_training_data():
    load_training_data("training_data")

test_load_training_data()