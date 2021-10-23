
SET_TYPE = 'test'
HOME_PATH = '/home/mihai'

GROUND_TRUTH_PATH = 'datasets/SHAD/Walk-{}/Annotations/'.format(SET_TYPE)
GROUND_TRUTH_PATH += '{}'

IMAGES_PATH = "datasets/leftImg8bit_trainvaltest/leftImg8bit/val/"
Json_ground_truth_path = "datasets/leftImg8bit_trainvaltest/gtBbox_cityPersons_trainval/gtBboxCityPersons/val"

# VIDEO_PATH = 'data/Walk-'+SET_TYPE+'/video/{}'
VIDEO_PATH = 'datasets/SHAD/Walk-'+SET_TYPE+'/video/{}'
output = "./results/"

default_model_dir = 'models'
default_model = 'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite'

default_labels_dir = 'labels'
default_labels = 'coco_labels.txt'

# intersection over union minimum
IOU_THRESHOLD = 0.5
CORAL_THRESHOLD = 0.5


