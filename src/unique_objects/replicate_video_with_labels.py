import sys
import argparse
import cv2
from os.path import join

def main(vid_file, outdir):
    frame_idx = 0
    frames = []
    vidcap = cv2.VideoCapture(vid_file)
    frame_width = int(vidcap.get(3))
    frame_height = int(vidcap.get(4))
    frame_size = (int(frame_width/3),int(frame_height/3))
    #frame_size = (frame_width/2,frame_height/2)
    fps = 1
    output = cv2.VideoWriter(join(outdir, 'output_video_from_file.avi'), cv2.VideoWriter_fourcc('M','J','P','G'), fps, frame_size)
    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if ret == False:
            break
        font = cv2.FONT_HERSHEY_SIMPLEX
        # org
        org = (50, 50)
        # fontScale
        fontScale = 1
        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 2

        # Resize the frame
        frame = cv2.resize(frame, frame_size)

        # Using cv2.putText() method
        frame = cv2.putText(frame, "FrameIdx %d"%(frame_idx), org, font, fontScale, color, thickness, cv2.LINE_AA)
        output.write(frame)
        frame_idx += 1
    vidcap.release()
    output.release()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Duplicate the video but with labels.")
    parser.add_argument("-V", "--video-file", dest="video_file", help="Video file", required=True)
    parser.add_argument("-O", "--output-dir", dest="outdir", help="Output directory", default="/tmp")
    args = parser.parse_args()
    main(args.video_file, args.outdir)

