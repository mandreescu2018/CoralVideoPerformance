import cv2
import os
import time
import numpy as np

from utils import newresultutils, groundtruthutils, boundingboxutils, inferencecoralutils, datautils
from utils.configutils import ConfigurationParser

configuration = ConfigurationParser("config/config.txt", "EPFL")
inference_result = newresultutils.InferenceRes(dataset=configuration.section, model=configuration.model_path)
data_utils = datautils.DataUtils(configuration)

def main():
    dict_videos = data_utils.get_video_files_dict()
    start = time.perf_counter()
    for item in dict_videos:
        check_video(item['data'], item['annotation'])
    end = time.perf_counter()
    print(f"Duration: {end - start} sec")
    inference_result.calculate_and_graph()


def check_video(video_file, annotation):
    inference_times = []
    # video_path = os.path.join(configuration.home_path, configuration.data_path + video_file +'.mp4')
    cap = cv2.VideoCapture(video_file)

    interpreter, inference_size = inferencecoralutils.initialize_interpreter(os.path.abspath(configuration.model_path))
    ground_truth = groundtruthutils.GroundTruthTxt(annotation)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame

        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        xmlPath = os.path.join(configuration.home_path, configuration.annotations_path + video_file)

        # ground_truth, file_name_for_metrics = groundtruthutils.get_annotation_by_video_frame(xmlPath, frame_number)
        f_name = os.path.basename(annotation)
        file_name_for_metrics = f_name.split(".")[0] + str(frame_number)

        ground_truth_items = list(filter(lambda x: x.frame_no == frame_number, ground_truth.gt_items))
        ground_truth_items = [item for item in ground_truth_items if item.out_of_frame == 0]


        if len(ground_truth_items) > 0:
            cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)

            objs, inference_time = inferencecoralutils.make_inference_on_frame(interpreter, cv2_im_rgb, configuration.coral_threshold, 0)
            inference_times.append(inference_time)

            height, width, channels = cv2_im.shape
            scale_x, scale_y = width / inference_size[0], height / inference_size[1]

            scalled_bboxes = []
            for item in objs:
                bbox = item.bbox.scale(scale_x, scale_y)
                BB = boundingboxutils.BoundingBoxItem(bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, item.score)
                scalled_bboxes.append(BB)

            inference_result.store_boxes(ground_truth_items, scalled_bboxes, file_name_for_metrics, scale=True)

    cap.release()
    cv2.destroyAllWindows()
    print(repr(inference_result))
    # inferences_durations = np.array(inference_times)
    # print(np.mean(inferences_durations))
    print('Mean inference time %.2f ms' % np.mean(np.array(inference_times)))


if __name__ == '__main__':
    main()


