import cv2
import os
import moviepy.video.io.ImageSequenceClip

base_path = "D:/Machine_Learning/Anomaly_Detection_Research/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test001/"
new_path = "D:/Machine_Learning/Anomaly_Detection_Research/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test001_jpg/"
frames_path = "D:/Machine_Learning/Anomaly_Detection_Research/Implementation/frames/"


def convert_tif_to_jpg():
    for infile in os.listdir(base_path):
        if infile[-3:] == 'tif':
            read = cv2.imread(base_path + infile)
            outfile = infile.split('.')[0] + '.jpg'
            cv2.imwrite(new_path + outfile, read, [int(cv2.IMWRITE_JPEG_QUALITY), 200])


def create_video(folder_path, fps=30):
    image_files = [os.path.join(folder_path, img)
                   for img in os.listdir(folder_path)
                   if img.endswith(".jpg")]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    if folder_path == frames_path:
        clip.write_videofile('my_video_gray.mp4')
    else:
        clip.write_videofile('my_video.mp4')


def create_video_opencv(image_folder, video_name):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


def format_name(i, frame_count):
    return str(i).zfill(len(str(frame_count)))


def colour_to_black():
    source = cv2.VideoCapture('crowd_coloured.avi')

    frame_count = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(frame_count):
        ret, img = source.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        file_name = format_name(i + 1, frame_count)
        cv2.imwrite(frames_path + f'{file_name}.jpg', gray)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    source.release()


def main():
    if len(os.listdir(new_path)) == 0:
        convert_tif_to_jpg()

    # create_video_opencv(new_path, video_name='my_video.avi')
    colour_to_black()
    create_video_opencv(frames_path, 'my_video_gray.avi')
    print("Complete")


if __name__ == "__main__":
    main()
