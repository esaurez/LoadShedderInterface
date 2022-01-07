from pathlib import Path

import numpy as np
import csv
import matplotlib.pyplot as plt
import json
import random
import pandas as pd

occurrencesDictionary = dict()
utility_threshold_array = dict()#[]
max_utility_threshold_array = []
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

#colors = [10,20,30,50,70,90,110,130,150,170,180,255]
colors = ['red', 'orange','yellow','spring_green','green','ocean_green','light_blue','blue','purple','magenta']


def assign_hsb_feature_to_eqv_classes(featureList, split_values, color = None):
    """
    input: list of features with count per colors, list of split values (to which class to assign the color)
    output: list of class-assignments, e.g., [1,0,0,1,0,0] if we had two colors and the feature list has low value per color, i.e. maps to the first class.

    # also needed as output: list of the split values per color. need to store them and feed them into the model.
    # where is the model stored? - utility.csv and utility_threshold.csv

    takes a features List that has count per color and splits it into classes according to small or high count.
    how to split is defined by split_values
    for now, we have frames of same size so we take absolut split_values
    """    
   

    color_position = colors.index(color)
    color_feature = featureList[color_position] # value of the color.
    # find the respective bin

    # enumerate class
    i = 0
    for i in range(0,len(split_values)-1): # len = 20. i nimmt also Werte von 0 bis 18 an, da upper bound bei der range raus is.
        if color_feature > split_values[i] and color_feature <= split_values[i+1]:
            class_ = i 
            return [class_]
    return [i] # return it as a list. 

    ### replaced 28.08.
    #number_of_classes = len(split_values)+1 # e.g. only one value: two classes
    #classified_features = [0] * ((len(featureList)-1) * number_of_classes) # for color-of-interest version. 
    #for i in range(len(featureList)-1): # einen Wert weniger nehmen. der letzte Wert sind "others" und somit egal.
    #    if len(split_values) > 1:
    #        sv = 0
    #        for j in range(0,len(split_values)):
    #            if featureList[i] > split_values[j]:
    #                sv = j
    #        classified_features[i*number_of_classes+sv] = 1 ### to check!! 
    #    else:
    #        if featureList[i]<split_values[0]:
    #            classified_features[i*number_of_classes]  = 1
    #        else:
    #            classified_features[i*number_of_classes+1] = 1
    #return classified_features

def assign_hsb_features_to_eqv_classes(observations, split_values, color):
    converted_observations = []
    for observation in observations:
        featureList = observation[0:-1]
        label = observation[-1]
        class_assignment = assign_hsb_feature_to_eqv_classes(featureList,split_values, color)
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
            occurrencesDictionary[key][0] += 1 # total
            occurrencesDictionary[key][1] += label # label sum
        else:
            occurrencesDictionary[key] = [1, label, 0.0] # total, label sum, utility
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


def normalize_utilities(normalized_occurencesDictionary, minUtility, percentileList, percent, utilityNormalizationUpperBound):
    output = dict()
    maxUtility = percentileList[percent]

    for key, value in normalized_occurencesDictionary.items():
        output[key] = value
        ut = value[2]
        value.append(ut)
        if ((maxUtility - minUtility) == 0):
            ut_normalized = 0
        else:
            ut_normalized = int(utilityNormalizationUpperBound * (ut - minUtility) / (maxUtility - minUtility))
            if ut_normalized > utilityNormalizationUpperBound:
                ut_normalized = utilityNormalizationUpperBound
        output[key][2] = ut_normalized 

    return output
        

def compute_utility_frequency_and_threshold_max(max_dict):
    # this already has a utility!
    
    max_aggregated = max_dict.groupby('max_utility').count()
    max_aggregated.reset_index(inplace = True)
    max_sorted = max_aggregated.sort_values(by=['max_utility'])
    max_sorted['ratio'] = max_sorted.label.cumsum() / sum(max_sorted.label)
    return max_sorted[['max_utility','ratio']].to_numpy()


def compute_utility_frequency_and_threshold_all_colors(all_color_occurences): # optionally I can call it with a subset of colors.
    """
    input: occurencesDictionary in probabilities (so pure counting)
    array to hold the utility thresholds--
    index represents the utility (ut), value represents the frequency/occurrence of the ut
    """
   
    df = pd.DataFrame(all_color_occurences).T # wirklich mit .T?


    df.reset_index(inplace = True)
    df.columns = ['class','total','match','normalized_ut','ut']

    df.sort_values(by =['normalized_ut'], inplace = True)

    df['ratio'] = df.total.cumsum() / sum(df.total)
   # output = df[['normalized_ut','ratio']].to_numpy()
    output = df[['ratio']].to_numpy()
    return output





def compute_utility_frequncy_and_threshold_probabilities(occurrencesDictionary):
    """
    input: occurencesDictionary in probabilities (so pure counting)
    array to hold the utility thresholds--
    index represents the utility (ut), value represents the frequency/occurrence of the ut
    """
   
    df = pd.DataFrame(occurrencesDictionary).T


    df.reset_index(inplace = True)
    df.columns = ['class','total','match','normalized_ut','ut']

    df.sort_values(by =['ut'], inplace = True)

    df['ratio'] = df.total.cumsum() / sum(df.total)
    #print(df)
    output = df[['ut','ratio']].to_numpy()
    return output

#    utility_threshold_array = dict()
#    utility_frequency_in_occ_dict = dict()


#    total_number_of_occurrences = 0

#    for key, value in occurrencesDictionary.items():
#        occ = value[0] # was steht an position 1?
#        ut = value[2]
      #  utility_frequency_in_occ_dict[ut] += 1
#        old = utility_frequency_in_occ_dict.get(ut,0)
#        utility_frequency_in_occ_dict[ut] = old + 1
        
#        #utility_threshold_array[ut] += occ
#        old = utility_threshold_array.get(ut,0) 
#        utility_threshold_array[ut] = old + occ
#        total_number_of_occurrences += occ



    # sort dict by keys (i.e. utility)
    # or just make it as a df and then use cumsum stuff.
    # then you also have an index ... 

 #   utility_threshold_array = np.cumsum(np.array(utility_threshold_array))
 #   print(utility_threshold_array)
 #   utility_threshold_array = [x / total_number_of_occurrences for x in utility_threshold_array.values()]

    #return utility_threshold_array, utility_frequency_in_occ_dict


def compute_utility_frequncy_and_threshold(occurrencesDictionary, utilityNormalizationUpperBound):
    """
    input: occurencesDictionary, but already normalized.
    array to hold the utility thresholds--
    index represents the utility (ut), value represents the frequency/occurrence of the ut
    """
    utility_threshold_array = [0] * (utilityNormalizationUpperBound + 1)
    utility_frequency_in_occ_dict = [0] * (utilityNormalizationUpperBound + 1)


    #utility_threshold_array = dict()
    #utility_frequency_in_occ_dict = dict()


    total_number_of_occurrences = 0

    for key, value in occurrencesDictionary.items():
        occ = value[0]
        ut = value[2]
        utility_frequency_in_occ_dict[ut] += 1
      #  old = utility_frequency_in_occ_dict.get(ut,0)
      #  utility_frequency_in_occ_dict[ut] = old + 1
        
        utility_threshold_array[ut] += occ
       # old = utility_threshold_array.get(ut,0) 
       # utility_threshold_array[ut] = old + occ
        total_number_of_occurrences += occ

    utility_threshold_array = np.cumsum(utility_threshold_array)
    utility_threshold_array = [x / total_number_of_occurrences for x in utility_threshold_array]
    return utility_threshold_array, utility_frequency_in_occ_dict


def save_utilities(path, occurrencesDictionary, normalized=False):
    # create the model directory if not exists
    Path(path).mkdir(parents=True, exist_ok=True)

    title='/utilities.csv'
    if normalized:
        title='/normalized_utilities.csv'

    with open(path + title, 'w') as f:
        for key, value in occurrencesDictionary.items():
            if normalized:
                f.write('%s, %s\n' % (key, value[2]))
            else:
                f.write('%s, %s\n' % (key, value[3]))

def save_split_values(path, split_values):
    Path(path).mkdir(parents=True, exist_ok=True)

    with open(path + '/split_values.txt','w') as f:
       json.dump(split_values, f)
    f.close()

def load_split_values(modelPath):
    split_values = dict()
    for color in colors:
        with open(modelPath + '/'+color+'/split_values.txt','r') as f:
            svs = json.load(f)
        f.close()
        split_values[color] = svs

    return split_values

def load_utilities(path, normalized = False, all_colors = False): # load them for all the colors.
    colors = ['red', 'orange','yellow','spring_green','green','ocean_green','light_blue','blue','purple','magenta']
    title = '/utilities.csv'
    if normalized:
        title = '/normalized_utilities.csv'
    if not all_colors:
        occurrencesDictionary = dict()
        for color in colors:
            occurrencesDictionary[color] = dict()
            with open(path + '/' + color + title, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for line in csv_reader:
                    if normalized:
                        occurrencesDictionary[color][line[0]] = int(line[1])
                    else:
                        occurrencesDictionary[color][line[0]] = float(line[1].strip())
    if all_colors:
        occurrencesDictionary_allcolors = dict()
        with open(path + '/' + 'all_colors' + title, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for line in csv_reader:
                   # occurrencesDictionary_allcolors[line[0]] = int(line[1])
                    occurrencesDictionary_allcolors[line[0]] = float(line[1].strip())
        return occurrencesDictionary_allcolors
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


def save_utility_threshold(path, utility_threshold_array, normalized=False):
    # create the model directory if not exists
    Path(path).mkdir(parents=True, exist_ok=True)

    title = "/utility_threshold.csv"
    if normalized:
        title = "/utility_threshold_normalized.csv"

    with open(path + title, 'w') as f:
        if normalized:
            for i in range(0, len(utility_threshold_array)):
                f.write('%d, %f\n' % (i, utility_threshold_array[i]))
        else:
            for ut,ratio in utility_threshold_array:
                f.write(f'{ut},{ratio}\n')
            


def load_utility_threshold(path, color = None, normalized = False, all_colors = False):
    title = "/utility_threshold.csv"
    if color is None:
        if all_colors:
           # title = "/all_colors/utility_threshold_normalized.csv"
            title = "/all_colors/utility_threshold.csv"
            utility_threshold_array = []
            with open(path + title, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for line in csv_reader:
                    ut = line[0]
                    ratio = line[1]
                    utility_threshold_array.append((float(ratio), float(ut)))
                 #   utility_threshold_array.append((float(line[1])))
        else:
            utility_threshold_array = []
            with open(path + title, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for line in csv_reader:
                    ut = line[0]
                    ratio = line[1]
                    utility_threshold_array.append((float(ratio), float(ut)))
    else:
        if normalized:
            title = "/utility_threshold_normalized.csv"
        utility_threshold_array = []
        with open(path + '/'+color+title, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for line in csv_reader:
                ut = line[0]
                ratio = line[1]
                if normalized: 
                    utility_threshold_array.append((float(line[1])))
                else:
                    utility_threshold_array.append((float(ratio), float(ut)))

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

def rearrange_split_values(observations, min_bin_size, split_values):
    total = len(observations)
    min_count_per_class= total * min_bin_size # a bin needs at least min_bin_size*100 % of the data.
    df = pd.DataFrame(observations)
    df.columns = ['class_','label']

    counts_per_class = df.groupby('class_').count()

    split_values_reduced = []
    for i in range(len(split_values)):
        if i in counts_per_class.index.tolist():
            split_values_reduced.append(split_values[i]) # so here, only add those values that have a class anyways.
    split_values_reduced.append(split_values[-1]) # upper bound just needed for later.
   # last_value = split_values_reduced[-1]
   # last_value_index = split_values.index(last_value)
   # split_values_reduced.append(split_values[last_value_index+1])
    # now I have as many split values as classes. because I have the 0 as split-value, too. so the lower bound for every class. 
    while (counts_per_class[counts_per_class.label <= min_count_per_class].size) > 0:
        # take min:
        class_ = counts_per_class['label'].idxmin()
        count =counts_per_class.loc[counts_per_class['label'].idxmin()].label
        available_classes = counts_per_class.index.tolist()
        available_classes.sort() # you never know

        if count < min_count_per_class:
            lower = -1
            upper = -1
            class_index = available_classes.index(class_)
            if class_index > 0:
                lower_class= available_classes[class_index -1]
                lower = counts_per_class.loc[lower_class].label
            if class_index < len(available_classes)-1:
                upper_class= available_classes[class_index +1]
                upper = counts_per_class.loc[upper_class].label
            lower_split_value = split_values[class_] # class is the position in the original array.
            
            upper_split_value = split_values_reduced[split_values_reduced.index(lower_split_value)+1]

            
            if lower <= upper and lower != -1 or upper  == -1: # or upper == -1: Pick smaller group
          #  if lower >= upper and upper != -1 or upper  == -1: # or upper == -1: Pick bigger group
                lower_split_value = split_values[class_]
                split_values_reduced.remove(lower_split_value)
                df['class_'] = df['class_'].apply(lambda x: lower_class if x  == class_ else x)
            else:
              # merge upper class to lower. 
                df['class_'] = df['class_'].apply(lambda x: class_ if x  == upper_class else x)
                lower_split_value = split_values[class_]
                index_ = split_values_reduced.index(lower_split_value)
                del split_values_reduced[index_ + 1]
        counts_per_class = df.groupby('class_').count()

   # print(split_values_reduced)
    # for each split value, get the position in the original list.
    # this is then the mapping.
    mapping = dict()
    for new_pos, sv in enumerate(split_values_reduced):
        pos_org = split_values.index(sv)
        mapping[pos_org] = new_pos
    df['class_'] = df.class_.apply(lambda x: mapping[x])
    return(df.values.tolist(), split_values_reduced)

"""
build the model.
observations: list of list. It contains all gathered observations.
generatedModelPath: output path for storing the trained model
utilityNormalizationUpperBound: 
split_values: initial bin split values
min_bin_size: min fraction of training samples that should exist in each bin
featureCorrespondingBinSize: list. bin size for each feature in observations
"""
def train(observations,  generatedModelPath, utilityNormalizationUpperBound, init_bin_width = 0.1,min_bin_size = 0.1, featureCorrespondingBinSize=None):

    split_values=[]
    i = 0
    while i < 100:
        split_values.append(i)
        i += init_bin_width
    split_values.append(100)

    #discretize_observations(observations, featureCorrespondingBinSize) # rgb based

    # for loop over colors.
    classified_observations = dict()# todo.
    utilities = dict()
    label_added = False
    for color in colors:
        print("Processing color ", color)

        # Assigning each frame to original bins
        color_observations = assign_hsb_features_to_eqv_classes(observations,split_values, color)
        #print (split_values)

        # Recomputing bins to account for min bin size
        reduced_observations, output_split_values = rearrange_split_values(color_observations, min_bin_size, split_values)
        print (output_split_values)

        occurrencesDictionary = get_occurrence_count(reduced_observations) # hier habe ich ja eine tabelle mit label - color.
        # reduced observation: list of lists.
        classified_observations[color] = [x[0] for x in reduced_observations]
        
        #occurrencesDictionary = get_occurrence_count(observations)

        # alles was mit occurencesDictionary zu tun hat ruhig color-wise machen.
        # aber dann benötige ich noch ein utility mapping für jeden frame.
        # dh. ich mach ein utility look up für jeden frame und nehme das maximum. aber das geht erst, nachdem ich alle klassifiziert habe
        build_utility_array(occurrencesDictionary)
        min_occ, max_occ, min_ut, max_ut = find_min_and_max_utilities(occurrencesDictionary)
        percentile_list = get_percentile_utility_occurrences(occurrencesDictionary)

        # hier nimm die max-row vom occurences dictionary und geh ab dort weiter.

    
        normalized_occurencesDictionary = normalize_utilities(occurrencesDictionary.copy(), min_ut, percentile_list, 95, utilityNormalizationUpperBound)

        # save utilities
        save_utilities(f'{generatedModelPath}/{color}', normalized_occurencesDictionary, True)
        save_utilities(f'{generatedModelPath}/{color}', occurrencesDictionary) # holds the probabilities.
        utilities[color] = occurrencesDictionary
        save_split_values(f'{generatedModelPath}/{color}', output_split_values)

        # for debugging
        plot_occurrences(f'{generatedModelPath}/{color}', normalized_occurencesDictionary, max_occ)

        utility_threshold_array_normalized, utility_frequency_in_occ_dict = compute_utility_frequncy_and_threshold(
        normalized_occurencesDictionary, utilityNormalizationUpperBound)

        # threshold utilities for non_normalized data
        #     
        utility_threshold_array = compute_utility_frequncy_and_threshold_probabilities(occurrencesDictionary)

        save_utility_threshold(f'{generatedModelPath}/{color}', utility_threshold_array_normalized, True)
        save_utility_threshold(f'{generatedModelPath}/{color}', utility_threshold_array)

        # for debugging
        plot_utility_threshold(f'{generatedModelPath}/{color}', utility_threshold_array_normalized, utility_frequency_in_occ_dict)

    #### after computing color wise utilities, for each frame, find the max and store the joined distribution. ### 
    # now replace classified observations with utilities.

    classified_observations['label'] = [x[1]for x in reduced_observations] # add the label at the very end. simply take the last "reduced observation"
    classified_observations_df = pd.DataFrame(classified_observations)
    classified_observations_utilities = pd.DataFrame(classified_observations)

    classified_observations_utilities['max_color'] = classified_observations_utilities.apply(lambda x : get_color_utility(x,utilities)[1], axis = 1)
    classified_observations_utilities['max_utility'] = classified_observations_utilities.apply(lambda x : get_color_utility(x,utilities)[0], axis = 1)


    # now count those utilites and sort them and store them
    # aggregate utilities.

    # compute utilities for max of utility
    max_cdf = compute_utility_frequency_and_threshold_max(classified_observations_utilities)
    save_utility_threshold(f'{generatedModelPath}', max_cdf)

    # compute utilities for all classs together
    
    # if only a subset of the colors shall be used, select them from the "classified_observations_df" accordingly.
    # some formatting:
    classified_observations_array = classified_observations_df.values
    all_color_occurences = get_occurrence_count(classified_observations_array) # dict with keys as string, total count, label, utility
    # now compute utility per class.
    build_utility_array(all_color_occurences) # dict with key, label, utility 
    min_occ, max_occ, min_ut, max_ut = find_min_and_max_utilities(all_color_occurences)
    percentile_list_all_colors = get_percentile_utility_occurrences(all_color_occurences)

    
    #print("all colors utility distribution:")
    #print(percentile_list_all_colors) # activate again for diff. ds
    

        # hier nimm die max-row vom occurences dictionary und geh ab dort weiter.
    
    normalized_all_color_occurences = normalize_utilities(all_color_occurences.copy(), min_ut, percentile_list_all_colors, 95, utilityNormalizationUpperBound)

    utility_threshold_array_array_colors = compute_utility_frequncy_and_threshold_probabilities(all_color_occurences)
   # print(utility_threshold_array_array_colors)

    
        # save utilities
    save_utilities(f'{generatedModelPath}/all_colors', normalized_all_color_occurences, True)
    save_utilities(f'{generatedModelPath}/all_colors', all_color_occurences)

        # for debugging
    plot_occurrences(f'{generatedModelPath}/all_colors', all_color_occurences, max_occ)

    #utility_threshold_array_normalized, utility_frequency_in_occ_dict = compute_utility_frequncy_and_threshold(
    #    all_color_occurences, utilityNormalizationUpperBound)


    utility_threshold_array_normalized, utility_frequency_in_occ_dict = compute_utility_frequncy_and_threshold(
    normalized_occurencesDictionary, utilityNormalizationUpperBound)

        # threshold utilities for non_normalized data
        #     
    utility_threshold_array = compute_utility_frequncy_and_threshold_probabilities(all_color_occurences)

    save_utility_threshold(f'{generatedModelPath}/all_colors', utility_threshold_array_normalized, True)
    save_utility_threshold(f'{generatedModelPath}/all_colors', utility_threshold_array)    

    # so for all colors, I use the /all_colors_utility_threshold_normalized.csv and the "normalized_utilities.csv" 

    # store all_colors_cdf. 

    save_utility_threshold(f'{generatedModelPath}/all_colors', utility_threshold_array_normalized, True)


    
    

def get_color_utility(row, utilities):
    max_utility = 0
    max_color = ''
    for color in colors: 
        class_ = row.loc[color]
        utility = utilities[color][f"'{class_}'"][3]
        if utility > max_utility:
            max_utility = utility
            max_color = color
    return (max_utility, max_color)


def get_utility_threshold(dropRatio, mode):
    '''
    returns first the utility threshold below which to shed, then the ratio.
    '''
    if mode == 'max_cdf':
        for ratio, ut in max_utility_threshold_array:
            if ratio >= dropRatio:
                return ut, ratio

    if mode == 'all_colors':
        for ratio, ut in all_color_utility_threshold_array:
            if ratio >= dropRatio:
                return ut, ratio
    if mode == 'random':
        return dropRatio, dropRatio # utility = ratio. 
    # to be used with normalized 
     #   for i,value  in enumerate(all_color_utility_threshold_array):
     #       print("all colors utility threshold:")
     #       print(value)
     #       if value >= dropRatio:
     #           return i, value
     #   return 100,1

    # deprecated
    # if normalized:
    #    for i in range(len(utility_threshold_array[color])):
        # works for normalized, only
    #        if (utility_threshold_array[color][i] >= dropRatio):
    #            return i, utility_threshold_array[color][i]

    #    return len(utility_threshold_array[color]),100
    #else: 
    #    for ratio, ut in utility_threshold_array[color]:
    #        if ratio >= dropRatio:
    #            return ut, ratio
    #    return 1,1



def get_utility(featureList, mode):
    utility = 0
    if mode == 'random':
        utility = random.random() # returns random no between 0 and 1
      
    if mode == 'all_colors':
        # get the class for each color.
        # merge them together and get the key from this class assignment.
        class_assignments = []
        for color in colors:
            class_assignment = assign_hsb_feature_to_eqv_classes(featureList, split_values[color], color) # takes all split values
            class_assignments.append(class_assignment[0])
        key = get_observation_key(class_assignments)
        utility =  all_colors_occurencesDictionary.get(key,len(all_color_utility_threshold_array)) # returns the max! not the min utility now. 
    if mode =='max_cdf':
        max_utility = 0
        for color in colors: 
            color_split_values = split_values[color]
            #featureList = assign_hsb_feature_to_eqv_classes(featureList, split_values, color)
            class_assignment = assign_hsb_feature_to_eqv_classes(featureList, color_split_values, color)
        

        # 1. load split values into dict
        # featureList = discretize_observation(featureList, featureCorrespondingBinSize)
        # key for features (except label)
            key = get_observation_key(class_assignment)

        # get utility, if not found, return the max utility

    #        utility = occurrencesDictionary[color].get(key, len(utility_threshold_array))
            utility = occurrencesDictionary[color].get(key, 0)
            if utility > max_utility:
                max_utility = utility
                max_color = color

        utility = max_utility
    return utility #, max_color, key


def init_shedding(modelPath):#, featureBinSize):
  #  global featureCorrespondingBinSize
  #  featureCorrespondingBinSize = featureBinSize
    global split_values
    split_values = load_split_values(modelPath)
    global occurrencesDictionary
    occurrencesDictionary = load_utilities(modelPath)
    global all_colors_occurencesDictionary 
    # all_colors_occurencesDictionary= load_utilities(modelPath, normalized=True, all_colors=True)
    all_colors_occurencesDictionary= load_utilities(modelPath, normalized=False, all_colors=True)
    global utility_threshold_array
    global max_utility_threshold_array
    global all_color_utility_threshold_array
    for color in colors:
        utility_threshold_array[color] = load_utility_threshold(modelPath, color)
       # utility_threshold_array = load_utility_threshold(modelPath)
    max_utility_threshold_array = load_utility_threshold(modelPath) # color = None means load overall max array
    all_color_utility_threshold_array = load_utility_threshold(modelPath, all_colors=True)

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
