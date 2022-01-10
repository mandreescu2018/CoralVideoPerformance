import tflite_runtime.interpreter as tflite
import time
from PIL import Image
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from pycoral.adapters import common

from pycoral.adapters import detect
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

from utils import newresultutils, groundtruthutils, boundingboxutils
from utils.configutils import ConfigurationParser
# from config import config

configuration = ConfigurationParser("config/config.txt", section='Cityscape')

inference_result = newresultutils.InferenceRes(dataset='Cityscape', model=configuration.model_path)

def main():
    start = time.perf_counter()

    # interpreter = make_interpreter(configuration.model_path)
    interpreter = tflite.Interpreter(model_path=configuration.model_path)
    interpreter.allocate_tensors()
    inference_size = common.input_size(interpreter)

    images_path = os.path.join(configuration.home_path, configuration.data_path)
    annotations_path = os.path.join(configuration.home_path, configuration.annotations_path)
    images, jsons = groundtruthutils.get_data(images_path, annotations_path)

    for index_img in range(len(images)):
        image = Image.open(images[index_img])

        json_file = jsons[index_img]
        ground_truth_annotations = groundtruthutils.GroundTruthJSON(json_file)
        if len(ground_truth_annotations.gt_items) == 0:
            continue

        # image.resize(inference_size)
        # run_inference(interpreter, image.tobytes())
        # objs = detect.get_objects(interpreter, configuration.coral_threshold)

        _, scale = common.set_resized_input(
            interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))
        interpreter.invoke()
        objs = detect.get_objects(interpreter, configuration.coral_threshold, scale)

        # interest for persons only
        objs = [item for item in objs if item.id == 0]

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
    inference_result.calculate_and_graph()


if __name__ == '__main__':
    main()

