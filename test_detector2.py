import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow_hub as hub
import cv2
import os
import time
import numpy
import tensorflow as tf
import pandas as pd
from utils import inferenceutils, pathutils
from pathlib import Path

detector = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1")
# detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")
# detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
# detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1")
# detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite3/detection/1")
# detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite3/detection/1")

labels = pd.read_csv('labels/labels.csv', sep=';', index_col='ID')
labels = labels['OBJECT (2017 REL.)']

# cap = cv2.VideoCapture('/home/mihai/datasets/SHAD/Walk-test/video/SJTU-SEIEE-170_Walk_0006.mp4')

inference_size=(512, 512)
width = 512
height = 512

from utils import newresultutils, groundtruthutils, boundingboxutils
from utils.configutils import ConfigurationParser
configuration = ConfigurationParser("config/config.txt", section='SHAD')
inference_result = newresultutils.InferenceRes(dataset='SHAD', model='efficientdet/lite3/detection/1')


def get_video_files():
    files_path = os.path.join(configuration.home_path, configuration.data_path.format(''))
    files = os.listdir(files_path)
    video_files_dict = {i: files[i].rsplit('.', 1)[0] for i in range(0, len(files))}
    return video_files_dict


def main():
    dict_videos = get_video_files()
    start = time.perf_counter()
    # check_video('SJTU-SEIEE-96_Walk_0014')
    for key, value in dict_videos.items():
        check_video(value)
    end = time.perf_counter()
    print(f"Duration: {end - start} sec")
    inference_result.calculate_and_graph()


def check_video(video_file):
    video_path = os.path.join(configuration.home_path, configuration.data_path + video_file + '.mp4')
    # video_path = pathutils.reconstruct_broken_string(video_path)
    if not os.path.exists(video_path):
        print("file not found")
        exit(0)

    cap = cv2.VideoCapture(video_path)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            cap.release()
            cv2.destroyAllWindows()
            break

        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
        xmlPath = os.path.join(configuration.home_path, configuration.annotations_path + video_file)
        xmlpath = Path(xmlPath)

        file_name = video_file + '-{}.xml'

        f_name = file_name.format(str(int(frame_number)).zfill(4))
        file_name_for_metrics = f_name.split(".")[0]
        f_name = os.path.join(xmlpath, f_name)
        f_name = f_name.replace(os.sep, '/')

        ground_truth = groundtruthutils.GroundTruthXml(f_name)

        # # Resize to respect the input_shape
        # frame = cv2.resize(frame, inference_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        output_dict = inferenceutils.run_inference_for_opencvframe(detector, frame)
        # boxes, scores, classes, num_detections
        # if isinstance(output_dict, tuple):
        #     print('tuple')

        # boxes = output_dict[0]
        # classes = output_dict[2]
        # scores = output_dict[1]

        boxes = output_dict["detection_boxes"]
        # boxes *= 100
        classes = output_dict["detection_classes"]
        scores = output_dict["detection_scores"]

        pred_labels = classes.numpy().astype('int')[0]
        # pred_labels = [labels[i] for i in pred_labels if i == 1]
        # pred_labels = [labels[i] for i in pred_labels]
        pred_boxes = boxes.numpy()[0].astype('int')
        pred_scores = scores.numpy()[0]

        # loop throughout the detections and place a box around it
        img_boxes = np.array([])
        scalled_bboxes = []
        for score, (ymin, xmin, ymax, xmax), label in zip(pred_scores, pred_boxes, pred_labels):
            if (score < 0.2) or (label > 1):
                continue

            BB = boundingboxutils.BoundingBoxItem(xmin, ymin, xmax, ymax, score)
            scalled_bboxes.append(BB)

            score_txt = f'{round(100 * score, 0)}'
            img_boxes = cv2.rectangle(frame, (xmin, ymax), (xmax, ymin), (0, 255, 0), 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_boxes, labels[label], (xmin, ymax - 10), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img_boxes, score_txt, (xmax, ymax - 10), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        if ground_truth.gt_items:
            # for item in ground_truth.gt_items:
            #     img_boxes = cv2.rectangle(img_boxes, (item.bbox.xmin, item.bbox.ymax), (item.bbox.xmax, item.bbox.ymin), (0, 0, 255), 1)
            inference_result.store_boxes(ground_truth.gt_items, scalled_bboxes, file_name_for_metrics, scale=True)

        # Display the resulting frame

        if img_boxes.any():
            cv2.imshow('black and white', img_boxes)
        else:
            cv2.imshow('black and white', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    print(repr(inference_result))


if __name__ == '__main__':
    main()