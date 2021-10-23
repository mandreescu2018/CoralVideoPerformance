from utils import resultutils
import os
from config import config
global inference_result
global home_path


def init():
    global inference_result
    inference_result = resultutils.InferenceResult()
    global home_path
    home_path = config.HOME_PATH