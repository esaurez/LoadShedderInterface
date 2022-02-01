import argparse
import pandas as pd
import numpy as np
import matplotlib
from matplotlib.font_manager import FontProperties
matplotlib.use("Agg")
import matplotlib.pyplot as plt

breakdown_data = {}

def add_data(component, latency):
    if latency == None:
        print (component, latency)
        exit(0)
    if component not in breakdown_data:
        breakdown_data[component] = []
    breakdown_data[component].append(latency)

def main(log, outdir):
    frame_data = {"cvtColor":None, "diff":None, "fastHsv":None}
    with open(log) as f:
        for line in  f.readlines():
            if "Time for " not in line:
                continue

            s = line.split()
            metric = s[-2]
            val = int(s[-1][1:-1])

            if metric in frame_data:
                frame_data[metric] = val
            if metric == "FeatureExtractorNode::ProcessFrame":
                for k in frame_data:
                    add_data(k, frame_data[k])
                    frame_data[k] = None

    medians = {}
    for k in breakdown_data:
        medians[k] = [np.median(breakdown_data[k])]
    
    df = pd.DataFrame.from_dict(medians)
    df.rename(columns={"cvtColor": 'Color Space Transformation', "diff": 'Background Subtraction', "fastHsv": "Feature Extraction"}, inplace=True)

    fontsize=12
    plt.close()
    fig,ax = plt.subplots(figsize=(8,2))
    df.plot.barh(stacked=True, ax=ax)
    #df.plot(kind="bar", stacked=True, ax=ax)
    fontP = FontProperties()
    fontP.set_size(fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize, bottom=False)
    legend = ax.legend(fancybox=True,shadow=False, prop=fontP)
    ax.set_xlabel("Component Latency [ms]", fontsize=fontsize)
    fig.savefig("barchart.png", bbox_inches="tight")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", dest="log", help="Path to the log file output by Rewind", required=True)
    parser.add_argument("-O", dest="outdir", help="Path to the output dir")
    args = parser.parse_args()
    main(args.log, args.outdir)
