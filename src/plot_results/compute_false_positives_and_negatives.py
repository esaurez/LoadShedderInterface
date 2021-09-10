from typing import List
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import argparse
from pathlib import Path

def get_share_of_useful_frames(video_path):
    try:
        data_frame = pd.read_csv(video_path, header=0, delimiter=",")
    except:
        print("problem in file: " + video_path)

    # data_frame["false_negatives"] = np.where(data_frame['shed_decision'] is True & data_frame['useful_gt'] is True, 1, 0)
    useful_frames = data_frame[data_frame['useful_gt'] == True].count()['useful_gt']
    total_frames = data_frame.count()['useful_gt']
    
    share_of_useful_frames = int(useful_frames * 100 // total_frames)

    # what we want is the number of shed frames and the number of those that have been shed but are positive now.
    # number of false negatives / number of shed events.
    return share_of_useful_frames

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

 # 1000 frames
 # 10 thereof are matches


 # shed 10% --> shed 100 100/1000 --> 10% 
 # 1 matches in the dropped data 


 #  1/10 --> 1% 
 #  1 / 100 --> 1%: ratio of matches in the whole ds 
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
        false_negatives_percentage = get_false_negatives(result_file) # total matches
        false_negative_from_shed_data = get_false_negative_from_shed_data (result_file)
        useful_events_share = get_share_of_useful_frames(result_file) # should be the same for all datasets.
        shedding_ratio = get_shedding_ratio(result_file)
        better_than_random = ((shedding_ratio * 100 - false_negatives_percentage)/(shedding_ratio*100))*100 #(40-40)/40 
        video_name = os.path.splitext(os.path.basename(result_file))[0]
        row = [video_name, false_negatives_percentage, false_negative_from_shed_data, useful_events_share, better_than_random, shedding_ratio]
        result_list.append(row)

    cols = ["video_name", "false_negatives", "false_negatives_in_shed_data", "useful_events_share","better_than_random","shedding_ratio"]
    result_data_frame = pd.DataFrame(result_list, columns=cols)

    return result_data_frame

# deprecated
def plot_mode_lines(data_frame, result_directory, y_axis, title_y_axis, figure_title, show_ratios = False):
    fig1, axes1 = plt.subplots()
    data_frame = data_frame[data_frame['mbs'] == '0.2'] # per bin size
    ratios = data_frame.ratio.unique()
    # plot by min_bin_size.
    mode_list = data_frame['mode'].unique()
    # make one line per mbs.
    for mode in mode_list:
        df = data_frame[data_frame['mode'] == mode ]
        df.plot(x='shedding_ratio', y=y_axis, ax = axes1, label=mode, marker='x')


    if show_ratios:
        for ratio in ratios:
            axes1.axvline(float(ratio), color="lightgrey", linestyle="dashed")

    axes1.legend(loc='best', prop={'size': 12})
    axes1.set_xlabel('Shedding Ratio', fontsize=14)
    axes1.tick_params(axis='x', labelsize=12)

    axes1.tick_params(axis='y', labelsize=12)
    axes1.set_ylabel(title_y_axis, fontsize=14)
    axes1.set_ylim(bottom=-1, top=100)

    axes1.set_title(figure_title)

    fig1.savefig(result_directory + f'/{title_y_axis}.png', bbox_inches='tight')
    #fig1.savefig('false-negatives.pdf', bbox_inches='tight')
    plt.close()


def plot_mbs_lines(data_frame, result_directory, y_axis, title_y_axis, figure_title, show_ratios = False, random=True):
    # plots usually false negatives by shedding rates for different minimum bin sizes and random if desired.
    fig1, axes1 = plt.subplots()
    df_random = data_frame[data_frame['mode'] == 'random']
    df_random = df_random.groupby('ratio').mean()
    data_frame = data_frame[data_frame['mode'] == 'max_cdf']
    ratios = data_frame.ratio.unique()
    # plot by min_bin_size.
    mbs_list = data_frame.mbs.unique()
    # make one line per mbs.
    mbs_list = ['0.2','0.3'] 
    markers = ['x','v']

    for i,mbs in enumerate(mbs_list):
        df = data_frame[data_frame['mbs'] == mbs ]
        df.plot(x='shedding_ratio', y=y_axis, ax = axes1, label=mbs, marker=markers[i])
    
    if random:
        df_random.reset_index(inplace=True)
        df_random.plot(x='shedding_ratio',y=y_axis, ax = axes1, label='random', marker='o')


    if show_ratios:
        for ratio in ratios:
            axes1.axvline(float(ratio), color="lightgrey", linestyle="dashed")

    axes1.legend(loc='best', prop={'size': 12})
    axes1.set_xlabel('Actual shedding rate', fontsize=18)
    axes1.tick_params(axis='x', labelsize=12)

    axes1.tick_params(axis='y', labelsize=12)
    axes1.set_ylabel(title_y_axis, fontsize=18)
    axes1.set_ylim(bottom=-1, top=100)

   # axes1.set_title(figure_title)


    fig1.savefig(result_directory + f'/{title_y_axis}.png', bbox_inches='tight')
    #fig1.savefig('false-negatives.pdf', bbox_inches='tight')
    plt.close()


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


# deprecated
def plot_bars_by_mode(data_frame, result_directory, mbs,y = 'false_negatives', x='ratio',title = 'false_negatives'):
    """
    creates a bar chart with one bar per mode, ratios on x and fn on the y axis. 
    """
    fig1, axes1 = plt.subplots()
    
    n = len(pd.unique(data_frame[x]))
     # no of available ratios
    ind = [float(x_value) for x_value in pd.unique(data_frame[x])]

    width = np.min(np.diff(ind))/3
    
    df1 = data_frame[data_frame['mode'] =='all_colors']
    df2 = data_frame[data_frame['mode']=='max_cdf']

    ind1 = [float(x_value) for x_value in df1[x]]#df1[x].values #float(x_value) for x_value in pd.unique(df1[x])]
    ind2 = [float(x_value) for x_value in df2[x]]# #[float(x_value) for x_value in pd.unique(df2[x])]

    y1 = df1[y].values
    y2 = df2[y].values

    axes1.bar(ind1 - width/2, y1, width,color='seagreen', label='all colors')
    axes1.bar(ind2 + width/2,y2, width, label='max utility')
    
    axes1.legend(loc='best', prop={'size': 12})
    axes1.set_xlabel(x, fontsize=14)
  #  axes1.axes.set_xticklabels(pd.unique(data_frame.ratio))
    axes1.tick_params(axis='x', labelsize=12)

    #axes1.tick_params(axis='y', labelsize=12)
    axes1.set_ylabel(y, fontsize=14)
    axes1.set_ylim(bottom=-1, top=100)
    axes1.set_title(f'Min sample-share per bin: {mbs} off all samples')

    fig1.savefig(result_directory + f'/{mbs}_{title}.png', bbox_inches='tight')

    

def plot_fn(data_frame, result_directory,):
    # shows how min bin sizes behave for different shedding ratios.
    ratios = ['0.3','0.5','0.7']
    data_frame = data_frame[data_frame['mode'] == 'max_cdf'][['mbs','false_negatives', 'ratio', 'shedding_ratio']]
    fig1, axes1 = plt.subplots()
    
    df = data_frame.pivot(index='mbs', columns='ratio', values='false_negatives')
    df.plot.bar(y=ratios, ax = axes1)
   # df.plot(bar(x='mbs'), y=)
        
  #  for ratio in ratios:
   # for ratio in data_frame.ratio.unique():
   #     data_frame[data_frame['ratio'] == ratio].plot.line(x='mbs', y='false_negatives', ax=axes1, label = ratio)

    axes1.legend(loc='best', prop={'size': 12})
    axes1.set_xlabel('Minimum bin size', fontsize=18)
    axes1.tick_params(axis='x', labelsize=12)

    axes1.tick_params(axis='y', labelsize=12)
    axes1.set_ylabel("% False negatives", fontsize=18)
    axes1.set_ylim(bottom=-1, top=100)
  #  axes1.set_title("False negatives over mbs for different shedding rates")

    fig1.savefig(result_directory + f'/false-negatives_over_mbs.png', bbox_inches='tight')
    plt.close()
    #fig1.savefig('false-negatives.pdf', bbox_inches='tight')

def plot_ratios(data_frame, result_directory):
    # shows deviation of required rate from achieved rate per mbs.
    data_frame = data_frame[data_frame['mode'] == 'max_cdf']
    data_frame.ratio = data_frame.ratio.astype(float)
    print(data_frame.dtypes)
    data_frame['ratio_delta'] = ((data_frame['shedding_ratio']- data_frame['ratio'])) #/data_frame['ratio'])*100
   
    # plot by ratio
    fig1, axes = plt.subplots(2, 1, sharex=True)
    print(axes)
    for i,mbs in enumerate(['0.2','0.3']): #data_frame.mbs.unique():
        print(data_frame[data_frame['mbs']==mbs]['ratio_delta'])
        data_frame[data_frame['mbs']==mbs].plot.bar(x='ratio', y='ratio_delta', ax=axes[i], label=mbs)

    #axes.legend(loc='best', prop={'size': 12})
    axes[1].set_xlabel('Required rate', fontsize=18)
    
   # axes[0].tick_params(axis='x', labelsize=12)
    #axes[0].tick_params(axis='y', labelsize=12)
    #axes[0].set_ylabel("Actual rate - required rate", fontsize=14)
    fig1.text(0.02, 0.5, "Deviation of actual from reqd. rate", va='center', rotation='vertical', fontsize=16)
    axes[0].set_ylim(bottom=-0.2, top=0.5)
    axes[1].set_ylim(bottom=-0.2, top=0.5)

    #quired rate over mbs")

    axes[0].label_outer()
    axes[1].label_outer()
    fig1.savefig(result_directory + f'/ratios.png', bbox_inches='tight')
    plt.close()

    
def plot_ratios_by_mbs(data_frame, result_directory):
    # shows how the achieved shedding ratio deviates from the required over mbs. --> compare mbs performance.
    data_frame = data_frame[data_frame['mode'] == 'max_cdf']
    data_frame.ratio = data_frame.ratio.astype(float)
    print(data_frame.dtypes)
    data_frame['ratio_delta'] = (data_frame['shedding_ratio']- data_frame['ratio']) #/data_frame['ratio']
    
    #fig1, axes1 = plt.subplots()

    # plot by ratio

    ratios = [0.3, 0.5, 0.7]
    fig1, axes1 = plt.subplots()
    df = data_frame.pivot(index='mbs', columns='ratio', values='ratio_delta')
    df.plot.bar(y=ratios, ax = axes1)

    
    axes1.legend(loc='best', prop={'size': 12})
    axes1.set_xlabel('Minimum bin size', fontsize=18)
    axes1.tick_params(axis='x', labelsize=12)

    axes1.tick_params(axis='y', labelsize=12)
    axes1.set_ylabel("Deviation of actual from reqd. rate", fontsize=18)
    axes1.set_ylim(bottom=-0.2, top=0.5)
   # axes1.set_title("Deviation from required rate over mbs for different shedding rate")
    fig1.savefig(result_directory + f'/ratios_by_mbs.png', bbox_inches='tight')
    plt.close()


"""
Plot the false postives and negatives given in the data frame
"""

# deprecated
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
        
    plot_mbs_lines(df,f'{result_directory}/plots','false_negatives','% False negatives ', 'FN by minimum bin-size')
    plot_mbs_lines(df,f'{result_directory}/plots','better_than_random','% Quality improvement over random', 'Performance of max-utility approach by mbs', show_ratios=False, random=False)
    
    plot_fn(df,f'{result_directory}/plots')
    plot_ratios_by_mbs(df,f'{result_directory}/plots')

    plot_ratios(df,f'{result_directory}/plots')

    # not used
   # plot_mbs_lines(df,f'{result_directory}/plots','false_negatives_in_shed_data','% False Negatives in Shed Data', 'FN in shed events by minimum bin-size ')
   # plot_mbs_lines(df,f'{result_directory}/plots','false_negatives','% False Negatives, Required Rate', 'FN by minimum bin-size', show_ratios=True)
   # plot_mbs_lines(df,f'{result_directory}/plots','false_negatives_in_shed_data','% False Negatives in Shed Data, Required Rate', 'FN in shed events by minimum bin-size ', show_ratios=True)
    #plot_mode_lines(df,f'{result_directory}/plots','better_than_random','% Improvement over Random', 'Performance of max-utility Approach', show_ratios=False)
    #plot_fn(df, f'{result_directory}/plots')
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