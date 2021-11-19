import argparse
import cv2
from concurrent.futures import ProcessPoolExecutor
from os.path import join
import concurrent
import read_pixel_hsv_from_video

def write_hsv(frame_idx, frame, frame_dir):
    outfile = join(frame_dir, "frame_%d.txt"%frame_idx)
    with open(outfile, "w") as f:
        for row in range(len(frame)):
            for col in range(len(frame[row])):
                (h,s,v) = frame[row][col]
                if h == 0 and s == 0 and v == 0:
                    continue
                f.write("%d\t%d\t%d\n"%(h,s,v))

def main(vid_file, outdir):
    executor = ProcessPoolExecutor(max_workers=32)

    frame_dir = outdir
    frame_idx = 0
    futures = []
    for frame in read_pixel_hsv_from_video.get_frames(vid_file):
        future = executor.submit(write_hsv, frame_idx, frame, frame_dir)
        futures.append(future)
        frame_idx += 1

    done = 0
    for f in concurrent.futures.as_completed(futures):
        done+=1
        print ("Done with %d frame"%done)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-V", "--video-file", dest="video_file", help="Path to video file")
    parser.add_argument("-O", dest="outdir", help="Directory to store output plots")

    args = parser.parse_args()
    main(args.video_file, args.outdir)

