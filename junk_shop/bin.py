def rescale_bbox(cv2_im, inference_size, objs):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        obj.bbox = obj.bbox.scale(scale_x, scale_y)

MODEL_PATH = models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite
MODEL_PATH = models/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite