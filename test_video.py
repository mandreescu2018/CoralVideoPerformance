import cv2
import os
import time

import pycoral
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import input_size
from pycoral.utils.edgetpu import run_inference

from pathlib import Path
from utils import xmlutils, resultutils, metricutils, newresultutils
from config import config
from config import settings

# video_name = 'SJTU-SEIEE-175_Walk_0002'
# video_name = 'SJTU-SEIEE-96_Walk_0014'
video_name = 'SJTU-SEIEE-177_Walk_0039'

settings.init()
# Res = resultutils.InferenceResult()
inf_res = newresultutils.InferenceRes()

file_name = video_name + '-{}.xml'

def compare_boxes(boxes_ground_truth, boxes_prediction):
    boxes_prediction_cpy = boxes_prediction.copy()
    boxes_ground_truth_cpy = boxes_ground_truth.copy()

    matches_true_positives = []
    matches_false_positives = []

    total_results = len(boxes_prediction)
    ground_truth_total = len(boxes_ground_truth)
    settings.inference_result.add_ground_truths(len(boxes_ground_truth))
    true_positives = 0
    for elem_gt in boxes_ground_truth_cpy:
        matched = False
        for idx, elem_res in enumerate(boxes_prediction_cpy):
            iou = metricutils.bb_intersection_over_union(elem_gt.bbox, elem_res)
            if iou >= config.IOU_THRESHOLD:
                matched = True
                settings.inference_result.add_true_positives_number(1)
                del boxes_prediction_cpy[idx]
                break

        matches_true_positives.append(1 if matched else 0)

    settings.inference_result.add_true_positives(matches_true_positives)

def get_video_files():
    # home_path = os.path.expanduser('~')
    files_path = os.path.join(settings.home_path, config.VIDEO_PATH.format(''))
    files = os.listdir(files_path)
    dictFiles = {i: files[i].rsplit('.', 1)[0] for i in range(0, len(files))}
    # print(dictFiles[2])
    return dictFiles


def main():

    dict_videos = get_video_files()

    for key, value in dict_videos.items():
        main_run(value)
    inf_res.calculate_mean_average_precision()
    inf_res.calculate_and_graph()


def main_run(video_file):

    model = os.path.join(config.default_model_dir, config.default_model)
    video_path = os.path.join(settings.home_path, config.VIDEO_PATH.format(video_file +'.mp4'))
    cap = cv2.VideoCapture(video_path)

    labels = os.path.join(config.default_labels_dir, config.default_labels)

    labels = read_label_file(labels) if labels else {}
    interpreter = make_interpreter(model)
    interpreter.allocate_tensors()
    inference_size = input_size(interpreter)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame

        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)

        xmlPath = os.path.join(settings.home_path, config.GROUND_TRUTH_PATH.format(video_file))
        xmlpath = Path(xmlPath)

        file_name = video_file + '-{}.xml'

        f_name = file_name.format(str(int(frame_number)).zfill(4))
        file_name_for_metrics = f_name.split(".")[0]
        f_name = os.path.join(xmlpath, f_name)
        f_name = f_name.replace(os.sep, '/')

        ground_truth_object = xmlutils.GroundTruth(f_name)

        if ground_truth_object.objects:
            cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
            run_inference(interpreter, cv2_im_rgb.tobytes())
            objs = detect.get_objects(interpreter, 0.4)[:11]

            cv2_im = append_objs_to_img(cv2_im, inference_size, objs, labels)
            cv2_im = append_objs_to_img(cv2_im, inference_size, ground_truth_object.objects, labels)

            height, width, channels = cv2_im.shape
            scale_x, scale_y = width / inference_size[0], height / inference_size[1]
            # scalled_bboxes = []
            # for item in objs:
            #     bbox = item.bbox.scale(scale_x, scale_y)
            #     scalled_bboxes.append(bbox)

            class BboxObj:
                pass

            scalled_bboxes = []
            for item in objs:
                bbox = item.bbox.scale(scale_x, scale_y)
                BB = BboxObj()
                setattr(BB, 'bbox', bbox)
                setattr(BB, 'id', item.id)
                setattr(BB, 'score', item.score)
                scalled_bboxes.append(BB)

            inf_res.store_boxes(ground_truth_object.objects, scalled_bboxes, file_name_for_metrics, scale=True)
            settings.inference_result.compare_boxes(ground_truth_object.objects, scalled_bboxes)

        cv2_im = cv2.putText(cv2_im, str(int(frame_number)).zfill(4), (10, 10 + 30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # encode the frame in JPEG format
        # (flag, encodedImage) = cv2.imencode(".jpg", cv2_im)
        # cv2.imshow('Video test', cv2_im)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()
    print(repr(settings.inference_result))
    # print(str(settings.inference_result))

    # settings.inference_result.precision_recall_curve()



def rescale_bbox(cv2_im, inference_size, objs):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        obj.bbox = obj.bbox.scale(scale_x, scale_y)


def append_objs_to_img(cv2_im, inference_size, objs, labels):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]

    for obj in objs:
        if obj.id != 0:
            continue
        if type(obj) == detect.Object:
            bbox = obj.bbox.scale(scale_x, scale_y)
            bounding_box_color = (0, 0, 255)
        else:
            bbox = obj.bbox
            bounding_box_color = (0, 255, 0)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), bounding_box_color, 2)

    return cv2_im


if __name__ == '__main__':
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(f"Duration: {end - start} sec")

