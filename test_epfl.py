import cv2
import os
import time
from utils.configutils import ConfigurationParser
from utils import inferenceutils, pathutils
import numpy as np
import pandas as pd

from pathlib import Path
from utils import newresultutils, groundtruthutils, boundingboxutils, configutils
from config import config
from config import settings
import tensorflow_hub as hub

labels = pd.read_csv('labels/labels.csv', sep=';', index_col='ID')
labels = labels['OBJECT (2017 REL.)']

detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")
# detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
# detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite3/detection/1")
# detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite3/detection/1")

configuration = ConfigurationParser("config/config.txt", "EPFL")
inference_result = newresultutils.InferenceRes(dataset='EPFL', model='efficientdet/lite3/detection/1')


# print(configuration.data_path)
# print(configuration.annotations_path)
# print(configuration.data_files)


def get_video_files():
    files = [os.path.abspath(os.path.join(configuration.data_path, p)) for p in os.listdir(configuration.data_path)]
    annotations = [os.path.abspath(os.path.join(configuration.annotations_path, p)) for p in
                   os.listdir(configuration.annotations_path)]
    video_files_dict = [{'data': files[i], 'annotation': annotations[i]} for i in range(0, len(files))]
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

        if ret:
            # Display the resulting frame
            ground_truth_items = list(filter(lambda x: x.frame_no == frame_number, ground_truth.gt_items))
            ground_truth_items = [item for item in ground_truth_items if item.out_of_frame == 0]
            # filtered_items = filter(lambda x: x.frame_no == frame_number, ground_truth.gt_items)

            f_name = os.path.basename(annotation)
            file_name_for_metrics = f_name.split(".")[0] + str(frame_number)

            output_dict = inferenceutils.run_inference_for_opencvframe(detector, frame)
            boxes = output_dict[0]
            classes = output_dict[2]
            scores = output_dict[1]

            pred_labels = classes.numpy().astype('int')[0]

            pred_boxes = boxes.numpy()[0].astype('int')
            pred_scores = scores.numpy()[0]
            # loop throughout the detections and place a box around it
            img_boxes = np.array([])
            scalled_bboxes = []
            for score, (ymin, xmin, ymax, xmax), label in zip(pred_scores, pred_boxes, pred_labels):
                if (score < 0.2) or (label > 1):  # only persons
                    continue

                BB = boundingboxutils.BoundingBoxItem(xmin, ymin, xmax, ymax, score)
                scalled_bboxes.append(BB)
                score_txt = f'{round(100 * score, 0)}'
                frame = cv2.rectangle(frame, (xmin, ymax), (xmax, ymin), (0, 0, 255), 1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, labels[label], (xmin, ymax - 10), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, score_txt, (xmax, ymax - 10), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

            for item in ground_truth_items:
                if item.out_of_frame:
                    continue
                x0, y0 = int(item.bbox.xmin), int(item.bbox.ymin)
                x1, y1 = int(item.bbox.xmax), int(item.bbox.ymax)

                frame = cv2.rectangle(frame, (x0, y0), (x1, y1), bounding_box_color, 2)

            cv2.imshow('Frame', frame)
            if ground_truth_items:
                # for item in ground_truth.gt_items:
                #     img_boxes = cv2.rectangle(img_boxes, (item.bbox.xmin, item.bbox.ymax), (item.bbox.xmax, item.bbox.ymin), (0, 0, 255), 1)
                inference_result.store_boxes(ground_truth_items, scalled_bboxes, file_name_for_metrics, scale=True)
                print(repr(inference_result))

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
    start = time.perf_counter()

    for item in dict_videos:
        check_video(item['data'], item['annotation'])
    end = time.perf_counter()
    print(f"Duration: {end - start} sec")
    inference_result.calculate_and_graph()


if __name__ == '__main__':
    # dct = get_video_files()
    # print(dct)
    main()
