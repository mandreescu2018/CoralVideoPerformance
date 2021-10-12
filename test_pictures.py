
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# import datautils as gt

import time
from PIL import Image
from PIL import ImageDraw
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

from utils import xmlutils, resultutils, metricutils, newresultutils, groundtruthutils, boundingboxutils

from config import config
from config import settings

# csv_obj = csvutils.CsvHelper("results.csv")
#
# Res = result.Result()
inf_res = newresultutils.InferenceRes()

def main():

    # labels = read_label_file(conf.labels)
    model = os.path.join(config.default_model_dir, config.default_model)
    interpreter = make_interpreter(model)
    interpreter.allocate_tensors()

    images, jsons = groundtruthutils.get_data(config.IMAGES_PATH, config.Json_ground_truth_path)
    # Res.number_of_images = len(images)

    for index_img in range(len(images)):
        image = Image.open(images[index_img])

        # cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        # cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        # run_inference(interpreter, cv2_im_rgb.tobytes())
        # objs = detect.get_objects(interpreter, 0.4)[:11]

        _, scale = common.set_resized_input(
            interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))

        # for _ in range(5):
        interpreter.invoke()
        objs = detect.get_objects(interpreter, config.CORAL_THRESHOLD, scale)

        objs = [item for item in objs if item.id == 0]

        boxes_result = []

        for item in objs:
            bbox = item.bbox
            BB = boundingboxutils.BoundingBoxItem(bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, item.score)
            BB.score = item.score
            boxes_result.append(BB)

        # boundingboxutils.BoundingBoxItem(bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, item.score)
        # gt.draw_objects(ImageDraw.Draw(image), objs, labels)

        json_file = jsons[index_img]
        ground_truth_annotations = groundtruthutils.GroundTruthJSON(json_file)
        if len(ground_truth_annotations.gt_items) == 0:
            continue

        # gt.draw_objects(ImageDraw.Draw(image), ground_truth_annotations.bounding_boxes)

        # compare_boxes(ground_truth_annotations.bounding_boxes, boxes_result, images[index_img])
        inf_res.store_boxes(ground_truth_annotations.gt_items, boxes_result, os.path.basename(images[index_img]), scale=True)

        # image.save(os.path.join(conf.output, 'result_coral_' + os.path.basename(images[index_img])))

    # result = met_iou.log_avg_mr_reference_implementation(Res)
    # print("result", result)
    inf_res.calculate_mean_average_precision()
    inf_res.calculate_and_graph()


if __name__ == '__main__':
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print("duration: {} sec".format(end - start))
    # print(Res)
