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


def calculate_false_positives_and_negatives(result_directory):
    result_list = []

    result_files = glob.glob(os.path.join(result_directory, "*.csv"))
    for result_file in result_files:
        false_negatives_percentage = get_false_negatives(result_file)
        video_name = os.path.splitext(os.path.basename(result_file))[0]
        row = [video_name, false_negatives_percentage, None]
        result_list.append(row)

    cols = ["video_name", "false_negatives", "false_positives"]
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


def process_false_positives_and_negative(result_directory):
    result_data_frame = calculate_false_positives_and_negatives(result_directory)
    plot(result_data_frame, result_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--result_directory", dest="result_directory",
                        help="The directory where the shedding results are stored", type=str, required=True)

    args = parser.parse_args()

    process_false_positives_and_negative(args.result_directory)
