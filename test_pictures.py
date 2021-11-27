
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from PIL import Image
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.edgetpu import make_interpreter

from utils import newresultutils, groundtruthutils, boundingboxutils

from config import config

inference_result = newresultutils.InferenceRes()

def main():

    model = os.path.join(config.default_model_dir, config.default_model)
    interpreter = make_interpreter(model)
    interpreter.allocate_tensors()

    images, jsons = groundtruthutils.get_data(config.IMAGES_PATH, config.Json_ground_truth_path)

    for index_img in range(len(images)):
        image = Image.open(images[index_img])

        json_file = jsons[index_img]
        ground_truth_annotations = groundtruthutils.GroundTruthJSON(json_file)
        if len(ground_truth_annotations.gt_items) == 0:
            continue

        _, scale = common.set_resized_input(
            interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))

        # for _ in range(5):
        interpreter.invoke()
        objs = detect.get_objects(interpreter, config.CORAL_THRESHOLD, scale)

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
        # image.save(os.path.join(conf.output, 'result_coral_' + os.path.basename(images[index_img])))

    inference_result.calculate_and_graph()


if __name__ == '__main__':
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print("duration: {} sec".format(end - start))
