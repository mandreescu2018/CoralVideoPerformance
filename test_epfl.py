import cv2
import os
import time
from utils.configutils import ConfigurationParser
from utils import newresultutils, groundtruthutils, boundingboxutils

import pycoral
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import input_size
from pycoral.utils.edgetpu import run_inference

from pathlib import Path
from utils import newresultutils, groundtruthutils, boundingboxutils, configutils
from config import config
from config import settings


configuration = ConfigurationParser("config/config.txt", "EPFL")
# print(configuration.data_path)
# print(configuration.annotations_path)
# print(configuration.data_files)

def get_video_files():
    files = [os.path.abspath(os.path.join(configuration.data_path, p)) for p in os.listdir(configuration.data_path)]
    annotations = [os.path.abspath(os.path.join(configuration.annotations_path, p)) for p in os.listdir(configuration.annotations_path)]
    video_files_dict = [{'data': files[i], 'annotation':annotations[i]} for i in range(0, len(files))]
    return video_files_dict

def check_video(video_file, annotation):
    # print('video: ', video_file)
    # print('annotation', annotation)
    cap = cv2.VideoCapture(video_file)
    ground_truth = groundtruthutils.GroundTruthTxt(annotation)

    bounding_box_color = (0, 255, 0)  # green

    while cap.isOpened():
        ret, frame = cap.read()
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if ret == True:
            # Display the resulting frame

            filtered_items = filter(lambda x: x.frame_no == frame_number, ground_truth.gt_items)

            for item in filtered_items:
                if item.out_of_frame:
                    continue
                x0, y0 = int(item.bbox.xmin), int(item.bbox.ymin)
                x1, y1 = int(item.bbox.xmax), int(item.bbox.ymax)

                frame = cv2.rectangle(frame, (x0, y0), (x1, y1), bounding_box_color, 2)

            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            # Break the loop
        else:
            break

        # When everything done, release
        # the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def main():

    dict_videos = get_video_files()

    for item in dict_videos:
        check_video(item['data'], item['annotation'])
    # inference_result.calculate_mean_average_precision()
    # inference_result.calculate_and_graph(dataset='SHAD')

if __name__ == '__main__':
    # dct = get_video_files()
    # print(dct)
    main()
