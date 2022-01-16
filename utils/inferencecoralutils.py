import time
import pycoral
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import input_size
from pycoral.adapters import common
from pycoral.utils.edgetpu import run_inference
from PIL import Image


def initialize_interpreter(model_path):
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    inference_size = input_size(interpreter)
    return interpreter, inference_size


def make_inference_on_frame(interpreter, opencv_frame, coral_threshold, person_annotation_id):
    start = time.perf_counter()
    run_inference(interpreter, opencv_frame.tobytes())
    inference_time = time.perf_counter() - start
    objs = detect.get_objects(interpreter, coral_threshold)

    # for the moment we are interested in persons detection only
    objs = [item for item in objs if item.id == person_annotation_id]
    return objs, inference_time * 1000

def run_inference_on_picture(interpreter, image, coral_threshold, person_annotation_id=0):
    start = time.perf_counter()
    _, scale = common.set_resized_input(
        interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    objs = detect.get_objects(interpreter, coral_threshold, scale)

    # interest for persons only
    objs = [item for item in objs if item.id == person_annotation_id]
    return objs, inference_time * 1000
