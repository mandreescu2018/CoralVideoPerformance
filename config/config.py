
SET_TYPE = 'test'

GROUND_TRUTH_PATH = './data/Walk-{}/Annotations/'.format(SET_TYPE)
GROUND_TRUTH_PATH += '{}'

VIDEO_PATH = 'data/Walk-'+SET_TYPE+'/video/{}'
output = "./results/"

default_model_dir = 'models'
default_model = 'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite'

default_labels_dir = 'labels'
default_labels = 'coco_labels.txt'

# intersection over union minimum
IOU_THRESHOLD = 0.5


