from utils import resultutils
import os

global inference_result
global home_path


def init():
    global inference_result
    inference_result = resultutils.InferenceResult()
    global home_path
    home_path = os.path.expanduser('~')