from pathlib import Path
import numpy as np
import csv
import matplotlib.pyplot as plt

occurrencesDictionary = dict()
utility_threshold_array = []
featureCorrespondingBinSize = []

"""
generate a key from given features.
featureList: list of features
"""


def get_observation_key(featureList):
    key = "-".join(str(x) for x in featureList)
    if key == "":
        key = 0
    # I changed here to avoid excel from changing the key into date format
    else:
        key = "'" + key + "'"
    return key


"""
Perform binning.
observations: list of list. It contains all gathered observations.
featureCorrespondingBinSize: bin size for each feature in observations
split_values: for hsb: divides the COUNT into "many and few", e.g., based on expected min-car size
"""

colors = [10,20,30,50,70,90,110,130,150,170,180,255]
def assign_hsb_feature_to_eqv_classes(featureList, split_values):
    """
    input: list of features with count per colors, list of split values (to which class to assign the color)
    output: list of class-assignments, e.g., [1,0,0,1,0,0] if we had two colors and the feature list has low value per color, i.e. maps to the first class.

    takes a features List that has count per color and splits it into classes according to small or high count.
    how to split is defined by split_values
    for now, we have frames of same size so we take absolut split_values
    """    
    
    number_of_classes = len(split_values)+1 # e.g. only one value: two classes
    #classified_features = [0] * (len(colors) * number_of_classes)
    classified_features = [0] * (len(featureList) * number_of_classes)
    #for i in range(len(colors)):
    for i in range(len(featureList)):
        if len(split_values) > 1:
            sv = 0
            for j in range(0,len(split_values)):
                if featureList[i] > split_values[j]:
                    sv = j
            classified_features[i*number_of_classes+sv] = 1 ### to check!! 
         #   print(f'feature {featureList[i]} belongs to class {sv} which is the value {split_values[sv]}')
        else:
            if featureList[i]<split_values[0]:
                classified_features[i*number_of_classes]  = 1
            else:
                classified_features[i*number_of_classes+1] = 1
              #  print(f'big enough for color:{colors[i]}')
   # print(classified_features)
    return classified_features

def assign_hsb_features_to_eqv_classes(observations, split_values):
    converted_observations = []
    for observation in observations:
        featureList = observation[0:-1]
        label = observation[-1]
        class_assignment = assign_hsb_feature_to_eqv_classes(featureList,split_values)
        class_assignment.append(label)
        converted_observations.append(class_assignment)
    return converted_observations

def discretize_observation(featureList, featureCorrespondingBinSize):
    # iterate over all features
    for i in range(0, len(featureList)):
        # divide by the bin size and keep only the integer part
        # not a nice way to handle the case where featureCorrespondingBinSize has less items
        if (i < len(featureCorrespondingBinSize)):
            featureList[i] = int(featureList[i] // featureCorrespondingBinSize[i])
        else:
            featureList[i] = int(featureList[i] // featureCorrespondingBinSize[0])
    return featureList


def discretize_observations(observations, featureCorrespondingBinSize):
    # iterate over all observations
    for observation in observations:
        # Last value in an observation is the label
        featureList = observation[0:-1]
        discretize_observation(featureList, featureCorrespondingBinSize)
        for i in range(len(featureList)):
            observation[i] = featureList[i]

"""
Count the number of occurrences of each distinct observation and the number of complex events detected 
for each distinct observation.

observations: list of observations
return a dictionary that contains the occurrences of each distinct observation and the corresponding detect complex events 
"""


def get_occurrence_count(observations):
    occurrencesDictionary = dict()
    for observation in observations:
        
        featureList = observation[:-1]
        label = observation[-1]

        key = get_observation_key(featureList)
        if key in occurrencesDictionary:
            occurrencesDictionary[key][0] += 1
            occurrencesDictionary[key][1] += label
        else:
            occurrencesDictionary[key] = [1, label, 0.0]
    return occurrencesDictionary
#    print(occurrencesDictionary)
    return occurrencesDictionary


"""
build the utilites by dividing the number of complex event by the number of occurrence in occurrencesDictionary
"""


def build_utility_array(occurrencesDictionary):
    for key, value in occurrencesDictionary.items():
        occ = value[0]
        sum_labels = value[1]
        ut = sum_labels / occ
        occurrencesDictionary[key][2] = ut
    #print(occurrencesDictionary)


def find_min_and_max_utilities(occurrencesDictionary):
    # Minimum and Maximum values' (occurrences & utilities) Calculation
    min_occ = 1000000
    max_occ = 0
    min_ut = 1000000
    max_ut = 0.0

    for key, value in occurrencesDictionary.items():
        occ = value[0]
        ut = value[2]

        if occ < min_occ:
            min_occ = occ
        if occ > max_occ:
            max_occ = occ

        if ut < min_ut:
            min_ut = ut
        if ut > max_ut:
            max_ut = ut

    return min_occ, max_occ, min_ut, max_ut


def get_percentile_utility_occurrences(occurrencesDictionary):
    utility_list = []
    percentile_list = []
    for key, value in occurrencesDictionary.items():
        ut = value[2]
        utility_list.append(ut)

    percentiles = [x for x in range(0, 101)]
    percentile_list = np.percentile(utility_list, percentiles)
    return percentile_list


def normalize_utilities(occurrencesDictionary, minUtility, percentileList, percent, utilityNormalizationUpperBound):
    maxUtility = percentileList[percent]

    for key, value in occurrencesDictionary.items():
        ut = value[2]
        if ((maxUtility - minUtility) == 0):
            ut_normalized = 0
        else:
            ut_normalized = int(utilityNormalizationUpperBound * (ut - minUtility) / (maxUtility - minUtility))
            if ut_normalized > utilityNormalizationUpperBound:
                ut_normalized = utilityNormalizationUpperBound
        occurrencesDictionary[key][2] = ut_normalized


def compute_utility_frequncy_and_threshold(occurrencesDictionary, utilityNormalizationUpperBound):
    """
    array to hold the utility thresholds--
    index represents the utility (ut), value represents the frequency/occurrence of the ut
    """
    utility_threshold_array = [0] * (utilityNormalizationUpperBound + 1)
    utility_frequency_in_occ_dict = [0] * (utilityNormalizationUpperBound + 1)

    total_number_of_occurrences = 0

    for key, value in occurrencesDictionary.items():
        occ = value[0]
        ut = value[2]
        utility_frequency_in_occ_dict[ut] += 1
        utility_threshold_array[ut] += occ
        total_number_of_occurrences += occ

    utility_threshold_array = np.cumsum(utility_threshold_array)
    utility_threshold_array = [x / total_number_of_occurrences for x in utility_threshold_array]

    return utility_threshold_array, utility_frequency_in_occ_dict


def save_utilities(path, occurrencesDictionary):
    # create the model directory if not exists
    Path(path).mkdir(parents=True, exist_ok=True)

    with open(path + "/utilities.csv", 'w') as f:
        for key, value in occurrencesDictionary.items():
            f.write('%s, %s\n' % (key, value[2]))


def load_utilities(path):
    occurrencesDictionary = dict()
    with open(path + "/utilities.csv", 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for line in csv_reader:
            occurrencesDictionary[line[0]] = int(line[1])

    return  occurrencesDictionary

def get_frequency_of_occurrences(occurrencesDictionary, max_occ):
    occurrences = [0] * (max_occ + 1)
    for value in occurrencesDictionary.values():
        occ = value[0]
        occurrences[occ] += 1

    return occurrences


def plot_occurrences(path, occurrencesDictionary, max_occ):
    # create the model directory if not exists
    Path(path).mkdir(parents=True, exist_ok=True)

    occurrences = get_frequency_of_occurrences(occurrencesDictionary, max_occ)

    su = len(occurrences)
    cdf = np.cumsum(occurrences)
    cdf = cdf / cdf[len(cdf) - 1]
    plt.clf()
    plt.plot(range(su), cdf)
    #     plt.show()
    plt.ylabel('frequency cumulative')
    plt.xlabel('occurrences')
    plt.savefig(path + "/occurrences_cumulative.png")

    plt.clf()
    y_pos = range(len(occurrences))

    bar_width = 0.4
    limit = len(y_pos)
    if limit > 100:
        limit = 100
    plt.bar(y_pos[0:limit], occurrences[0:limit], bar_width, align='center', alpha=0.5)
    # plt.yticks(y_pos)
    plt.ylabel('frequency')
    plt.xlabel('occurrences')
    plt.savefig(path + "/occurrences.png")


def save_utility_threshold(path, utility_threshold_array):
    # create the model directory if not exists
    Path(path).mkdir(parents=True, exist_ok=True)

    # I changed here, was giving array index out of bound so I changed it to -1
    with open(path + "/utility_threshold.csv", 'w') as f:
        for i in range(0, len(utility_threshold_array)):
            f.write('%d, %f\n' % (i, utility_threshold_array[i]))


def load_utility_threshold(path):
    utility_threshold_array = []
    with open(path + "/utility_threshold.csv", 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for line in csv_reader:
            utility_threshold_array.append(float(line[1]))

    return utility_threshold_array


def plot_utility_threshold(path, utility_threshold_array, utility_frequency_in_occ_dict):
    # create the model directory if not exists
    Path(path).mkdir(parents=True, exist_ok=True)

    su = len(utility_threshold_array)
    plt.clf()
    plt.plot(range(su), utility_threshold_array)
    plt.ylabel('threshold')
    plt.xlabel('utility')
    # plt.show()
    plt.savefig(path + "/utility_threshold_cumulative.png")

    x_pos = range(len(utility_frequency_in_occ_dict))
    bar_width = 0.8
    plt.clf()
    plt.bar(x_pos, utility_frequency_in_occ_dict, bar_width, align='center', alpha=0.5)
    plt.ylabel('frequency')
    plt.xlabel('utility')
    # plt.show()
    plt.savefig(path + "/utility_frequency_in_occ_dict.png")


"""
build the model.
observations: list of list. It contains all gathered observations.
featureCorrespondingBinSize: list. bin size for each feature in observations
"""


def train(observations,  generatedModelPath, utilityNormalizationUpperBound, split_values = None,featureCorrespondingBinSize=None):
    #discretize_observations(observations, featureCorrespondingBinSize) # rgb based

    observations = assign_hsb_features_to_eqv_classes(observations,split_values) 

    occurrencesDictionary = get_occurrence_count(observations)
    build_utility_array(occurrencesDictionary)
    #print(occurrencesDictionary)
    min_occ, max_occ, min_ut, max_ut = find_min_and_max_utilities(occurrencesDictionary)
    percentile_list = get_percentile_utility_occurrences(occurrencesDictionary)
    normalize_utilities(occurrencesDictionary, min_ut, percentile_list, 95, utilityNormalizationUpperBound)

    # save utilities
    save_utilities(generatedModelPath, occurrencesDictionary)

    # for debugging
    plot_occurrences(generatedModelPath, occurrencesDictionary, max_occ)

    # threshold utilities
    utility_threshold_array, utility_frequency_in_occ_dict = compute_utility_frequncy_and_threshold(
        occurrencesDictionary, utilityNormalizationUpperBound)
    save_utility_threshold(generatedModelPath, utility_threshold_array)
    # for debugging
    plot_utility_threshold(generatedModelPath, utility_threshold_array, utility_frequency_in_occ_dict)

    # print(occurrencesDictionary)


def get_utility_threshold(dropRatio):
    for i in range(len(utility_threshold_array)):
        if (utility_threshold_array[i] >= dropRatio):
            return i

    return len(utility_threshold_array)


def get_utility(featureList):
    featureList = assign_hsb_feature_to_eqv_classes(featureList, split_values)
  #  featureList = discretize_observation(featureList, featureCorrespondingBinSize)

    # key for features (except label)
    key = get_observation_key(featureList)

    # get utility, if not found, return the max utility
    utility = occurrencesDictionary.get(key, len(utility_threshold_array))
    return utility


def init_shedding(modelPath,sv):#, featureBinSize):
  #  global featureCorrespondingBinSize
  #  featureCorrespondingBinSize = featureBinSize
    global split_values
    split_values = sv
    global occurrencesDictionary
    occurrencesDictionary = load_utilities(modelPath)
    global utility_threshold_array
    utility_threshold_array = load_utility_threshold(modelPath)

def test_hsb():
    observations = [[10,2,1,1],
                    [1,2,1,1]
    ]
    split_values = [3]
    generatedModelDirectory = "./"
    featureCorrespondingBinSize = [1]
    utilityNormalizationUpperBound = 100
    train(observations, featureCorrespondingBinSize,generatedModelDirectory,utilityNormalizationUpperBound,split_values)
    
def test():
    observations = [[1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 2, 2, 0],
                    [1, 1, 1, 0],
                    [1, 2, 2, 1],
                    [3, 1, 1, 1],
                    [5, 4, 3, 0],
                    [3, 1, 1, 1],
                    [7, 7, 7, 0],
                    [2, 5, 3, 1]
                    ]

    featureCorrespondingBinSize = [1]
    generatedModelDirectory = "./"
    utilityNormalizationUpperBound = 100
    train(observations, featureCorrespondingBinSize, generatedModelDirectory, utilityNormalizationUpperBound)

    # resultDict= dict()
    # resultDict["1-1-1"] = 66
    # resultDict["1-2-2"] = 50
    # resultDict["3-1-1"] = 100
    # resultDict["5-4-3"] = 0
    # resultDict["7-7-7"] = 0
    # resultDict["2-5-3"] = 100


if __name__ == "__main__":
    test()
