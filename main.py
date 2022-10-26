import cv2
import os
from argparse import ArgumentParser
import glob

frames_path = "D:/Machine_Learning/Anomaly_Detection_Research/Implementation/frames/"

"""
    flow: data set frames -> data set video -> optical flow video -> optical flow b&w frames -> optical flow b&w video
"""


def create_video_opencv(image_folder, video_name, dataset_video=None):
    if dataset_video:
        os.remove(dataset_video)

    images = [img for img in os.listdir(image_folder) if img.endswith(".tif")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


def format_name(i, frame_count):
    return str(i).zfill(len(str(frame_count)))


def colour_to_black(video_name):
    source = cv2.VideoCapture(video_name)

    frame_count = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(frame_count):
        ret, img = source.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        file_name = format_name(i + 1, frame_count)
        cv2.imwrite(frames_path + f'{file_name}.tif', gray)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    source.release()


def inference(args):
    if len(os.listdir(frames_path)) != 0:
        files = glob.glob('frames/.*')
        for f in files:
            os.remove(f)

    vid_name = args.video

    out_dir = args.output_dir
    if out_dir:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

    in_dir = args.input_dir
    if in_dir:
        create_video_opencv(in_dir, video_name=vid_name)  # creates video from dataset frames

    # TODO(integrate optical flow implementation)
    colour_to_black(video_name=vid_name)  # creates b&w frames from optical flow video
    create_video_opencv(frames_path, video_name=f'{out_dir}/gray_{vid_name}',
                        dataset_video=f'{vid_name}')  # creates b&w video from optical frames


def main():
    # TODO(reduce flickering)

    # TODO(read paper 1, GANS, pytorch course)

    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="enter directory of frames")
    parser.add_argument("--output_dir", type=str, help="enter directory of output video")
    parser.add_argument("--video", type=str, default="my_video.avi", help="enter video name")

    args = parser.parse_args()
    inference(args)

    print("Complete")


if __name__ == "__main__":
    main()
