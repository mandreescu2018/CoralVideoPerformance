import cv2
import os
import time

import pycoral
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import input_size
from pycoral.adapters import common
from pycoral.utils.edgetpu import run_inference

from pathlib import Path
from utils import newresultutils, groundtruthutils, boundingboxutils
from utils.configutils import ConfigurationParser

configuration = ConfigurationParser("config/config.txt", "SHAD")
inference_result = newresultutils.InferenceRes(dataset='SHAD', model=configuration.model_path)

def get_video_files():
    files_path = os.path.join(configuration.home_path, configuration.data_path.format(''))
    files = os.listdir(files_path)
    video_files_dict = {i: files[i].rsplit('.', 1)[0] for i in range(0, len(files))}
    return video_files_dict

def main():
    dict_videos = get_video_files()
    start = time.perf_counter()
    for key, value in dict_videos.items():
        check_video(value)
    end = time.perf_counter()
    print(f"Duration: {end - start} sec")
    inference_result.calculate_and_graph()


def check_video(video_file):

    video_path = os.path.join(configuration.home_path, configuration.data_path + video_file +'.mp4')
    cap = cv2.VideoCapture(video_path)

    interpreter = make_interpreter(configuration.model_path)
    interpreter.allocate_tensors()
    inference_size = input_size(interpreter)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame

        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)

        xmlPath = os.path.join(configuration.home_path, configuration.annotations_path + video_file)
        xmlpath = Path(xmlPath)

        file_name = video_file + '-{}.xml'

        f_name = file_name.format(str(int(frame_number)).zfill(4))
        file_name_for_metrics = f_name.split(".")[0]
        f_name = os.path.join(xmlpath, f_name)
        f_name = f_name.replace(os.sep, '/')

        ground_truth = groundtruthutils.GroundTruthXml(f_name)

        if ground_truth.gt_items:
            cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
            run_inference(interpreter, cv2_im_rgb.tobytes())

            objs = detect.get_objects(interpreter, configuration.coral_threshold)

            # interest for persons only (see coco_labels.txt)
            objs = [item for item in objs if item.id == 0]

            height, width, channels = cv2_im.shape
            scale_x, scale_y = width / inference_size[0], height / inference_size[1]

            scalled_bboxes = []
            for item in objs:
                bbox = item.bbox.scale(scale_x, scale_y)
                BB = boundingboxutils.BoundingBoxItem(bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, item.score)
                scalled_bboxes.append(BB)

            inference_result.store_boxes(ground_truth.gt_items, scalled_bboxes, file_name_for_metrics, scale=True)

    cap.release()
    cv2.destroyAllWindows()
    print(repr(inference_result))


if __name__ == '__main__':
    main()


