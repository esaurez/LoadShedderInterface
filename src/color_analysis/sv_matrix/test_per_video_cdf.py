import argparse
import pandas as pd

def compute_global_util_threshold(global_cdf, drop_rate):
    for idx in range(len(global_cdf)):
        (ratio, util) = global_cdf[idx]
        if ratio > drop_rate:
            prev = global_cdf[idx-1]
            slope = (util - prev[1])/(ratio - prev[0])

            util_threshold = prev[1] + slope*(drop_rate - prev[0])
            return util_threshold
        elif drop_rate == ratio:
            return util

def main(frame_utils, training_fraction, global_cdf_file):
    df = pd.read_csv(frame_utils)
    num_training_frames = int(training_fraction*len(df))
    util_vals = []
    test_util_vals = []
    for index, row in df.iterrows():
        if row["frame_id"] < num_training_frames:
            util_vals.append(row["utility"])
        else:
            test_util_vals.append(row["utility"])
                
    # Now compute the CDF
    util_vals = sorted(util_vals)

    # Read the global CDF
    global_cdf = []
    with open(global_cdf_file) as f:
        for line in f.readlines():
            s = line.split()
            global_cdf.append((float(s[0]), float(s[1])))

    # Theese are the testing drop rates
    drop_rates = [0.25, 0.5, 0.75]
    
    print ("target_drop_rate\tpervid_drop_rate\tglobal_drop_rate")
    for drop_rate in drop_rates:
        util_threshold = util_vals[int(drop_rate*len(util_vals))]

        global_util_threshold = compute_global_util_threshold(global_cdf, drop_rate)

        actual_drop_rate = len([x for x in test_util_vals if x < util_threshold])/len(test_util_vals)
        global_drop_rate = len([x for x in test_util_vals if x < global_util_threshold])/len(test_util_vals)
        print (drop_rate, "\t", actual_drop_rate, "\t", global_drop_rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzing the distribution of Pixel Fraction")
    parser.add_argument("-U", dest="frame_utils", help="Path to per-video frame_utils.csv file")
    parser.add_argument("-R", dest="training_fraction", help="Fraction of frames to use for building the CDF. Rest will be used for testing the drop rate to util  threshold mapping.", type=float)
    parser.add_argument("-G", dest="global_cdf", help="path to global CDF (across all videos)")

    args = parser.parse_args()
    main(args.frame_utils, args.training_fraction, args.global_cdf)

