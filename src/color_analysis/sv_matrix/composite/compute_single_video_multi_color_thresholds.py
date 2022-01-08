import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def normalize_utils(df, training_fraction):
    utils = []
    for idx, row in df.iterrows():
        utils.append(row["utility"])

    num_training_points = int(training_fraction*len(utils))

    max_util = max(utils[:num_training_points])
    min_util = min(utils[:num_training_points])

    for idx in range(len(utils)):
        utils[idx] = (utils[idx]-min_util)/(max_util - min_util)

    return utils

def compute_util_threshold(training_points, drop_ratio):
        for idx in range(len(training_points)):
            num_dropped = idx+1
            dropped = float(num_dropped)/len(training_points)

            if dropped == drop_ratio:
                # Frame at idx SHOULD BE dropped
                print ("1st", idx)
                return training_points[idx]
            elif dropped > drop_ratio:
                # Frame at idx SHOULD NOT BE dropped
                print ("2nd", idx)
                curr = training_points[idx]
                prev = training_points[idx-1]

                curr_drop = dropped
                prev_drop = idx/float(len(training_points))

                slope = (curr-prev)/(curr_drop-prev_drop)

                target_util = prev + slope*(drop_ratio-prev_drop)
                return target_util

def plot_utils(color_utils, max_utils, colors):
    fig, ax = plt.subplots(figsize=(6,4))
    idx = 0
    for c in color_utils:
        ax.plot(c, label=colors[idx])
        idx +=  1
    #ax.plot(max_utils, label="max")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Utility")
    fig.legend()
    fig.savefig("utilities.png")

def main(frame_utils, training_fraction, drop_ratios, colors):
    max_utils = []
    color_utils = []
    for frame_util in frame_utils:
        df = pd.read_csv(frame_util)
        
        # Compute normalized utilities
        norm_utils = normalize_utils(df, training_fraction)
        color_utils.append(norm_utils)
        for idx in range(len(norm_utils)):
            if idx == len(max_utils):
                max_utils.append(norm_utils[idx])
            else:
                max_utils[idx] = max(max_utils[idx], norm_utils[idx])

    plot_utils(color_utils, max_utils, colors)
    
    num_training_points = int(training_fraction*len(max_utils))

    training_utils = []
    test_utils = []
    for idx in range(num_training_points):
        training_utils.append(max_utils[idx])
    for idx in range(num_training_points, len(max_utils)):
        test_utils.append(max_utils[idx])

    training_utils = sorted(training_utils)

    util_thresholds = []
    obs_drop_ratios = []
    for drop_ratio in drop_ratios:
        util_threshold = compute_util_threshold(training_utils, drop_ratio)
        obs_drop_ratio = len([x for x in test_utils if x < util_threshold])/len(test_utils)
        #print ("%.3f\t%.3f\t%.3f"%(drop_ratio, util_threshold, obs_drop_ratio))
        util_thresholds.append(util_threshold)
        obs_drop_ratios.append(obs_drop_ratio)

    plt.close()
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(drop_ratios, util_thresholds, color="blue")
    ax.set_ylabel("Utility Threshold")
    ax.set_xlabel("Target Drop Ratio")
    ax.yaxis.label.set_color("blue")
    ax2 = ax.twinx()
    ax2.plot(drop_ratios, obs_drop_ratios, color="red")
    ax2.set_ylabel("Observed Drop Ratio")
    ax2.yaxis.label.set_color("red")
    fig.savefig("util_thresholds_drops.png", bbox_to_inches="tight")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-U", dest="frame_utils", help="Path to per-video frame_utils.csv files, one for each color", required=True, nargs="+")
    parser.add_argument("-C", dest="colors", help="Colors", nargs="+")
    #parser.add_argument("-D", dest="drop_ratios", help="Drop ratios (multiple)", required=True, nargs="+", type=float)
    parser.add_argument("-R", dest="training_fraction", help="Fraction of frames to use for building the CDF. Rest will be used for testing the drop rate to util  threshold mapping.", type=float, required=True)

    args = parser.parse_args()

    drop_ratios = []
    d = 0.05
    while d < 1.0:
        drop_ratios.append(d)
        d += 0.05

    main(args.frame_utils, args.training_fraction, drop_ratios, args.colors)
