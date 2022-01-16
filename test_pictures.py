import time
from PIL import Image
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import newresultutils, groundtruthutils, boundingboxutils, datautils, inferencecoralutils
from utils.configutils import ConfigurationParser

configuration = ConfigurationParser("config/config.txt", section='Cityscape')
data_utils = datautils.DataUtils(configuration)
inference_result = newresultutils.InferenceRes(dataset='Cityscape', model=configuration.model_path)

def main():
    start = time.perf_counter()
    inference_times = []

    interpreter, inference_size = inferencecoralutils.initialize_interpreter(os.path.abspath(configuration.model_path))

    images, jsons = data_utils.get_data()

    for index_img in range(len(images)):
        image = Image.open(images[index_img])

        json_file = jsons[index_img]
        ground_truth_annotations = groundtruthutils.GroundTruthJSON(json_file)
        if len(ground_truth_annotations.gt_items) == 0:
            continue

        objs, inference_time = inferencecoralutils.run_inference_on_picture(interpreter, image,
                                                                           configuration.coral_threshold)
        inference_times.append(inference_time)

        boxes_result = []
        for item in objs:
            bbox = item.bbox
            BB = boundingboxutils.BoundingBoxItem(bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, item.score)
            BB.score = item.score
            boxes_result.append(BB)

        inference_result.store_boxes(ground_truth_annotations.gt_items, boxes_result, os.path.basename(images[index_img]), scale=True)
        print(repr(inference_result))

    end = time.perf_counter()
    print("duration: {} sec".format(end - start))
    print('Mean inference time %.2f ms' % np.mean(np.array(inference_times)))
    inference_result.calculate_and_graph()


if __name__ == '__main__':
    main()

