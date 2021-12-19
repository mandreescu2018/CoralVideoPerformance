import numpy as np
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import cv2

def run_inference_for_opencvframe(detector, frame):

    # Is optional but i recommend (float convertion and convert img to tensor image)
    rgb_tensor = tf.convert_to_tensor(frame, dtype=tf.uint8)
    # Add dims to rgb_tensor
    rgb_tensor = tf.expand_dims(rgb_tensor, 0)
    output_detector = detector(rgb_tensor)
    return output_detector

def run_inference_for_single_image(detector, image):
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    output_dict = detector(input_tensor)
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections


    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    if 'detection_masks' in output_dict:
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict

# def show_inference(model, image_path, class_id):
#     image_np = np.array(Image.open(image_path))
#     output_dict = run_inference_for_single_image(model, image_np)
#     boxes = []
#     classes = []
#     scores = []
#     for i, x in enumerate(output_dict['detection_classes']):
#         if x == class_id and output_dict['detection_scores'][i] > 0.5:
#             classes.append(x)
#             boxes.append(output_dict['detection_boxes'][i])
#             scores.append(output_dict['detection_scores'][i])
#     boxes = np.array(boxes)
#     classes = np.array(classes)
#     scores = np.array(scores)
#     vis_util.visualize_boxes_and_labels_on_image_array(
#         image_np,
#         boxes,
#         classes,
#         scores,
#         category_index,
#         instance_masks=output_dict.get('detection_masks_reframed', None),
#         use_normalized_coordinates=True,
#         line_thickness=2)
#
#     display(Image.fromarray(image_np))