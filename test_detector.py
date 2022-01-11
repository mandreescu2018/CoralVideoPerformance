
import tensorflow_hub as hub
import cv2
import numpy
import tensorflow as tf
import pandas as pd
from utils import inferenceutils


detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")
# detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
labels = pd.read_csv('labels/labels.csv', sep=';', index_col='ID')
labels = labels['OBJECT (2017 REL.)']

cap = cv2.VideoCapture('D:/datasets/Walk-test/video/SJTU-SEIEE-170_Walk_0006.mp4')

inference_size=(512, 512)
width = 512
height = 512


while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        cap.release()
        cv2.destroyAllWindows()
        break

    # # Resize to respect the input_shape
    frame = cv2.resize(frame, inference_size)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    output_dict = inferenceutils.run_inference_for_opencvframe(detector, frame)
    # boxes, scores, classes, num_detections
    # if isinstance(output_dict, tuple):
    #     print('tuple')
    boxes = output_dict[0]
    classes = output_dict[2]
    scores = output_dict[1]


    # boxes = output_dict["detection_boxes"]
    # boxes *= 100
    # classes = output_dict["detection_classes"]
    # scores = output_dict["detection_scores"]

    pred_labels = classes.numpy().astype('int')[0]
    pred_labels = [labels[i] for i in pred_labels]
    pred_boxes = boxes.numpy()[0].astype('int')
    pred_scores = scores.numpy()[0]

    # loop throughout the detections and place a box around it
    for score, (ymin, xmin, ymax, xmax), label in zip(pred_scores, pred_boxes, pred_labels):
        if score < 0.5:
            continue

        score_txt = f'{round(100 * score, 0)}'
        img_boxes = cv2.rectangle(frame, (xmin, ymax), (xmax, ymin), (0, 255, 0), 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_boxes, label, (xmin, ymax - 10), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img_boxes, score_txt, (xmax, ymax - 10), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('black and white', img_boxes)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()