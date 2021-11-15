VIDEO_DIR=~/LoadShedderInterface/data/seed_videos
util_files=$(find $VIDEO_DIR -name "frame_utils.csv")

python3 plot_frame_utils.py -U $util_files -O ./plots/
