import os
from pathlib import Path
from config import settings, config
import cv2
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import input_size
from pycoral.utils.edgetpu import run_inference

from junk_shop import resultutils, xmlutils


class VideoClipCollection:
    def __init__(self, video_file_list):
        self.video_file_list = video_file_list
        self.current_video = None
        self.current_video_index = 0
        self.initialize_video()


    def initialize_video(self):
        self.current_video = VideoClipCoral(self.video_file_list[self.current_video_index])

    def get_frame(self):
        frame = self.current_video.get_frame()
        if not frame:
            self.current_video_index += 1
            self.initialize_video()
        return self.current_video.get_frame()


class VideoClipCoral:
    def __init__(self, par_video_name, run_a_list=False):

        if not run_a_list:
            settings.inference_result.init_object()
        model = os.path.join(config.default_model_dir, config.default_model)
        labels = os.path.join(config.default_labels_dir, config.default_labels)

        self.labels = read_label_file(labels) if labels else {}
        self.interpreter = make_interpreter(model)
        self.interpreter.allocate_tensors()
        self.inference_size = input_size(self.interpreter)
        self.file_name = par_video_name + '-{}.xml'
        self.xmlpath = Path(config.GROUND_TRUTH_PATH.format(par_video_name))
        self.result_detail = resultutils.InferenceResult()

        # capturing video
        self.video = cv2.VideoCapture(config.VIDEO_PATH.format(par_video_name + '.mp4'))

    def append_objs_to_img(self, cv2_im, objs, labels, detection=True):
        for obj in objs:
            if obj.id != 0:
                continue

            bbox = obj.bbox
            bounding_box_color = (0, 255, 0)  # green
            if detection:
                bounding_box_color = (0, 0, 255)  # red

            x0, y0 = int(bbox.xmin), int(bbox.ymin)
            x1, y1 = int(bbox.xmax), int(bbox.ymax)

            cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), bounding_box_color, 2)

        return cv2_im

    def get_frame(self):

        while self.video.isOpened():
            objs = None
            ret, frame = self.video.read()
            if not ret:
                break

            cv2_im = frame

            frame_number = self.video.get(cv2.CAP_PROP_POS_FRAMES)

            f_name = self.file_name.format(str(int(frame_number)).zfill(4))
            f_name = os.path.join(self.xmlpath, f_name)
            f_name = f_name.replace(os.sep, '/')

            ground_truth_object = xmlutils.GroundTruth(f_name)

            if ground_truth_object.objects:
                cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
                cv2_im_rgb = cv2.resize(cv2_im_rgb, self.inference_size)
                run_inference(self.interpreter, cv2_im_rgb.tobytes())
                objs = detect.get_objects(self.interpreter, 0.4)[:11]

                height, width, channels = cv2_im.shape
                scale_x, scale_y = width / self.inference_size[0], height / self.inference_size[1]

                cv2_im = self.append_objs_to_img(cv2_im, self.get_scalled_bboxes(objs, scale_x, scale_y), self.labels)
                cv2_im = self.append_objs_to_img(cv2_im, ground_truth_object.objects, self.labels, detection=False)
                # compare_boxes(ground_truth_object.objects, scalled_bboxes)
                settings.inference_result.compare_boxes(ground_truth_object.objects, self.get_scalled_bboxes(objs, scale_x, scale_y))

            cv2_im = cv2.putText(cv2_im, str(int(frame_number)).zfill(4), (10, 10 + 30),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            # self.result_detail.compare_boxes(ground_truth_object.objects, objs)

            if cv2_im.any():
                ret, jpeg = cv2.imencode('.jpg', cv2_im)
            else:
                return None

            return jpeg.tobytes()

    def get_scalled_bboxes(self, objs, scale_x, scale_y):
        class BboxObj:
            pass
        scalled_bboxes = []
        for item in objs:
            bbox = item.bbox.scale(scale_x, scale_y)
            BB = BboxObj()
            setattr(BB, 'bbox', bbox)
            setattr(BB, 'id', item.id)
            scalled_bboxes.append(BB)
        return scalled_bboxes
