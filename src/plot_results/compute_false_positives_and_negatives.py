import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import argparse


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

    print(number_of_dropped_frames)
    print(number_of_total_frames)
    print(f'ratio = {number_of_dropped_frames / number_of_total_frames}')
    return number_of_dropped_frames / number_of_total_frames


def calculate_false_positives_and_negatives(result_directory):
    result_list = []

    result_files = glob.glob(os.path.join(result_directory, "*.csv"))
    for result_file in result_files:
        false_negatives_percentage = get_false_negatives(result_file)
        shedding_ratio = get_shedding_ratio(result_file)
        video_name = os.path.splitext(os.path.basename(result_file))[0]
        row = [video_name, false_negatives_percentage, None, shedding_ratio]
        result_list.append(row)

    cols = ["video_name", "false_negatives", "false_positives", "shedding_ratio"]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--result_directory", dest="result_directory",
                        help="The directory where the shedding results are stored", type=str, required=True)
    parser.add_argument("-t", "--total_file", dest="total_file",
                        help="The csv that collects from all configs", type=str, required=True)
    parser.add_argument("-bs", "--bin_size", dest="bin_size",
                        help="size per bin", type=int, required=True)
    parser.add_argument("-fbs", "--feature_bin_size", dest="feature_bin_size",
                        help="Scaling factor", type=int, required=True)
    parser.add_argument("-rt", "--ratio", dest="ratio",
                        help="Shedding ratio used", type=float, required=True)

    args = parser.parse_args()

    process_false_positives_and_negative(args.result_directory, args.total_file, args.bin_size, args.feature_bin_size, args.ratio)
