from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import argparse
from pathlib import Path


def get_false_negative_from_shed_data(video_path):
    try:
        data_frame = pd.read_csv(video_path, header=0, delimiter=",")
    except:
        print("problem in file: " + video_path)

    # data_frame["false_negatives"] = np.where(data_frame['shed_decision'] is True & data_frame['useful_gt'] is True, 1, 0)
    number_of_shed_frames = data_frame[data_frame['shed_decision'] == True].count()['shed_decision']
    number_of_false_negatives = data_frame[
        (data_frame['shed_decision'] == True) & (data_frame['useful_gt'] == True)].count()['useful_gt']

    false_negative_in_shed_events = int(number_of_false_negatives * 100 // number_of_shed_frames)

    # what we want is the number of shed frames and the number of those that have been shed but are positive now.
    # number of false negatives / number of shed events.
    return false_negative_in_shed_events

def get_false_negatives(video_path):
    try:
        data_frame = pd.read_csv(video_path, header=0, delimiter=",")
    except:
        print("problem in file: " + video_path)

    # data_frame["false_negatives"] = np.where(data_frame['shed_decision'] is True & data_frame['useful_gt'] is True, 1, 0)
    number_of_useful_frames = data_frame[data_frame['useful_gt'] == True].count()['useful_gt']
    number_of_false_negatives = data_frame[
        (data_frame['shed_decision'] == True) & (data_frame['useful_gt'] == True)].count()['useful_gt']

    false_negatives_percentage = int(number_of_false_negatives * 100 // number_of_useful_frames)

    # what we want is the number of shed frames and the number of those that have been shed but are positive now.
    # number of false negatives / number of shed events.

    print(len(data_frame.index))
    print("get total of useful frames")
    print(number_of_useful_frames)
    print(number_of_false_negatives)
    print(f"false neg percentage {false_negatives_percentage}")
    return false_negatives_percentage

def get_shedding_ratio(video_path):
    try:
        data_frame = pd.read_csv(video_path, header=0, delimiter=",")
    except:
        print("problem in file: " + video_path)

    number_of_dropped_frames = data_frame[data_frame['shed_decision'] == True].count()['shed_decision']
    number_of_total_frames = data_frame.count()['shed_decision']

  #  print(number_of_dropped_frames)
  #  print(number_of_total_frames)
   # print(f'ratio dropped to total frames= {number_of_dropped_frames / number_of_total_frames}')
    return number_of_dropped_frames / number_of_total_frames


def calculate_false_positives_and_negatives(result_directory):
    # print(f"searching for results in: {result_directory}")
    result_list = []

    result_files = glob.glob(os.path.join(result_directory, "*.csv"))


    for result_file in result_files:
        print(result_file)

        false_negatives_percentage = get_false_negatives(result_file)
        alse_negative_from_shed_data = get_false_negative_from_shed_data(result_file)
        shedding_ratio = get_shedding_ratio(result_file)
        video_name = os.path.splitext(os.path.basename(result_file))[0]
        row = [video_name, false_negatives_percentage, alse_negative_from_shed_data, shedding_ratio]
        result_list.append(row)

    cols = ["video_name", "false_negatives", "false_negatives_in_shed_data", "shedding_ratio"]
    result_data_frame = pd.DataFrame(result_list, columns=cols)

    return result_data_frame


def plot(data_frame, result_directory):
    fig1, axes1 = plt.subplots()

    data_frame.plot.bar(x='video_name', y='false_negatives', ax=axes1)

    axes1.legend(loc='best', prop={'size': 12})
    axes1.set_xlabel('video', fontsize=14)
    axes1.tick_params(axis='x', labelsize=12)

    axes1.tick_params(axis='y', labelsize=12)
    axes1.set_ylabel("% false negatives", fontsize=14)
    axes1.set_ylim(bottom=-1, top=100)

    fig1.savefig(result_directory + '/false-negatives.pdf', bbox_inches='tight')

def plot_fn_by_sr(data_frame, result_directory):
    fig1, axes1 = plt.subplots()

    data_frame.plot.bar(x='experiment', y='false_negatives', ax=axes1)

    axes1.legend(loc='best', prop={'size': 12})
    axes1.set_xlabel('Experiment', fontsize=14)
    axes1.tick_params(axis='x', labelsize=12)

    axes1.tick_params(axis='y', labelsize=12)
    axes1.set_ylabel("% false negatives", fontsize=14)
    axes1.set_ylim(bottom=-1, top=100)

    fig1.savefig(result_directory + '/false-negatives.png', bbox_inches='tight')
    #fig1.savefig('false-negatives.pdf', bbox_inches='tight')

def plot_ratio_by_sr(data_frame, result_directory):
    fig1, axes1 = plt.subplots()

    data_frame.plot.bar(x='experiment', y='shedding_ratio', ax=axes1)

    axes1.legend(loc='best', prop={'size': 12})
    axes1.set_xlabel('Experiment', fontsize=14)
    axes1.tick_params(axis='x', labelsize=12)

    axes1.tick_params(axis='y', labelsize=12)
    axes1.set_ylabel("Achieved Ratio", fontsize=14)
    axes1.set_ylim(bottom=0, top=1)
    axes1.axhline(y=0.5, color='r', linestyle='-')
    fig1.savefig(result_directory + '/ratios.png', bbox_inches='tight')
    #fig1.savefig('ratios.pdf', bbox_inches='tight')



def plot_fn_by_mode(data_frame, result_directory, mbs):
    fig1, axes1 = plt.subplots()
    
    n = len(pd.unique(data_frame.ratio))
     # no of available ratios
    ind = [float(x) for x in pd.unique(data_frame.ratio)]

    width = np.min(np.diff(ind))/3
    
    df1 = data_frame[data_frame['mode'] =='all_colors']
    df2 = data_frame[data_frame['mode']=='max_cdf']

    x1=df1['false_negatives'].values
    x2 = df2['false_negatives'].values

    axes1.bar(ind-width/2, x1, width,color='seagreen', label='all colors')
    axes1.bar(ind + width/2,x2, width, label='max utility')
    
    axes1.legend(loc='best', prop={'size': 12})
    axes1.set_xlabel('required ratio', fontsize=14)
  #  axes1.axes.set_xticklabels(pd.unique(data_frame.ratio))
    axes1.tick_params(axis='x', labelsize=12)

    #axes1.tick_params(axis='y', labelsize=12)
    axes1.set_ylabel("% false negatives", fontsize=14)
    axes1.set_ylim(bottom=-1, top=100)
    axes1.set_title(f'Min samples per bin: {mbs}% off all samples')

    fig1.savefig(result_directory + f'/{mbs}_false-negatives.png', bbox_inches='tight')

    

def plot_fn(data_frame, result_directory,):
    fig1, axes1 = plt.subplots()

    data_frame.plot.bar(x='ratio', y='false_negatives', ax=axes1)

    axes1.legend(loc='best', prop={'size': 12})
    axes1.set_xlabel('required ratio', fontsize=14)
    axes1.tick_params(axis='x', labelsize=12)

    axes1.tick_params(axis='y', labelsize=12)
    axes1.set_ylabel("% false negatives", fontsize=14)
    axes1.set_ylim(bottom=-1, top=100)

    fig1.savefig(result_directory + '/false-negatives.png', bbox_inches='tight')
    #fig1.savefig('false-negatives.pdf', bbox_inches='tight')

def plot_ratios(data_frame, result_directory):
    fig1, axes1 = plt.subplots()

    data_frame.plot.bar(x='ratio', y='shedding_ratio', ax=axes1)

    axes1.legend(loc='best', prop={'size': 12})
    axes1.set_xlabel('required ratio', fontsize=14)
    axes1.tick_params(axis='x', labelsize=12)

    axes1.tick_params(axis='y', labelsize=12)
    axes1.set_ylabel("achieved ratio", fontsize=14)
    axes1.set_ylim(bottom=0, top=1)

    fig1.savefig(result_directory + '/ratios.png', bbox_inches='tight')
    #fig1.savefig('ratios.pdf', bbox_inches='tight')



"""
Plot the false postives and negatives given in the data frame
"""


def process_false_positives_and_negative(result_directory, total_file, bin_size, feature_bin_size, ratio):
    result_data_frame = calculate_false_positives_and_negatives(result_directory)
    df = result_data_frame
    df['bin_size'] = bin_size
    df['feature_bin_size'] = feature_bin_size
    df['ratio'] = ratio

    df.to_csv(total_file, mode='a', header=False)
    plot(result_data_frame, result_directory)

def process_hsb_results(result_directory, ratios, minbinsizes,modes):
    df = pd.DataFrame()
    for ratio in ratios:
        for mbs in minbinsizes:
            for mode in modes:
                result_data_frame = calculate_false_positives_and_negatives(f'{result_directory}/{ratio}/{mbs}/{mode}')
                result_data_frame['ratio'] = ratio
                result_data_frame['mbs'] = mbs
                result_data_frame['mode'] = mode
                df = df.append(result_data_frame)
    
    print(df)
  
    Path(f'{result_directory}/plots').mkdir(parents=True, exist_ok=True)
    df.to_csv(f'{result_directory}/plots/results.csv') # store results that are then plotted

    for mbs in minbinsizes:
        plot_fn_by_mode(df[df.mbs == str(mbs)],f'{result_directory}/plots', mbs )
    #plot_fn(df, f'{result_directory}/plots')
    plot_ratios(df,f'{result_directory}/plots')
   # plot_ratios(df, '../test_results')


def process_hsb_results_by_splits(result_directory, splitvalues):
    df = pd.DataFrame()
    for experiment in range(int(splitvalues)):
        result_data_frame = calculate_false_positives_and_negatives(f'{result_directory}/{experiment}')
        result_data_frame['experiment'] = experiment
        df = df.append(result_data_frame)
    print(df)
    plot_fn_by_sr(df,result_directory)
    plot_ratio_by_sr(df,result_directory)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--result_directory", dest="result_directory",
                        help="The directory where the shedding results are stored", type=str, required=True)
    #parser.add_argument("-t", "--total_file", dest="total_file",
    #                    help="The csv that collects from all configs", type=str, required=False)
   # parser.add_argument("-bs", "--bin_size", dest="bin_size",
   #                     help="size per bin", type=int, required=True)
    #parser.add_argument("-fbs", "--feature_bin_size", dest="feature_bin_size",
    #                    help="Scaling factor", type=int, required=True)
    #parser.add_argument("-rt", "--ratio", dest="ratio",
    #                    help="Shedding ratio used", type=float, required=True)
    parser.add_argument("-sr", "--ratios", nargs='+', dest="ratios",
                        help="Shedding ratio used", required=False)#, type=List(float), required=True)
    parser.add_argument("-mbs", "--minbinsize", nargs='+', dest="minbinsize",
        help="percentage of samples per bin", required=False)#, type=List(float), required=True)
    parser.add_argument("-m", "--mode", nargs='+', dest="mode",
        help="all_class or max_cdf", required=False)#, type=List(float), required=True)

# Use like:
# python arg.py -l 1234 2345 3456 4567
    args = parser.parse_args()

    print(args)

    #process_false_positives_and_negative(args.result_directory, args.total_file, args.bin_size, args.feature_bin_size, args.ratio)
    process_hsb_results(args.result_directory, args.ratios, args.minbinsize, args.mode)
    #process_hsb_results_by_splits(args.result_directory, args.splitvalues)