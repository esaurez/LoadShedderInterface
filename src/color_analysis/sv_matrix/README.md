## Extract the frames of each video
`~/LoadShedderInterface/src/color_analysis/extract_per_frame_hsv.sh <training-conf>`
This script will process each video one by one. For each video, it translated each frame into a file containing a list of HSV values. The files corresponding to all frames of a given video are stores as `frames.tar.gz` in the directory of the video's data.

---
**NOTE**
This computation is embarassingly parallel, so high number of cores on the machine will speed up the process.
---

## Compute the SV Matrix of videos


## Computing utility of test frames

`./compute_test_frames_util.sh ../training_confs/red_only.yaml`

This Bash script calls a Python script for each video. 
`compute_test_frames_util.py -F <frame_dir> -C <training_conf> -B <bin_file> -O <video_dir> --bins <num_bins> --util-files <util_heatmap_files> -V <video>`
Note: The `<util_heatmap_files>` is a space separated list of files describing the utility value for each bin in the SV matrix for each color. The order of specifying the heatmap utility files must match the order in which colors are specified in the `<training_conf>` file.

By using the `-A` flag, the above script can ignore the train/test split of data and compute the utility values for all frames in the video.

This Python script generates a file called `frame_utils.csv` inside each video directory.

## Computing utility CDF
This is used for transforming Drop Rate into Utility Threshold.
Currently the CDF is computed over all the videos in the dataset and stored in the provided output directory. Along with it, per video CDF is also computed and stored in the directory of that video.

`python3 compute_utils_cdf.py -C ../training_confs/red_only.yaml -O /tmp/`

The script expects each video directory to have a file called `frame_utils.csv` which contains the utility value for each test frame in the video.

Since the file `frame_utils.csv` contains utility values for each color, the above script computes the maximum utility for each frame over all colors.
