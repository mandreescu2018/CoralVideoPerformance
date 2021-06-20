import numpy as np
from utils import metricutils
from config import config

class InferenceResult:
    def __init__(self):
        self.ground_truth_total = 0
        self._true_positives = []
        self.number_of_images = 0
        self._false_negatives = []
        self._false_positives = []
        self._false_positives_total = 0
        self._true_positives_total = 0

    @property
    def true_positives(self):
        true_positive_arr = np.array(self._true_positives)
        return np.cumsum(true_positive_arr)

    @property
    def false_positives(self):
        print("self.false_positives =>", self._false_positives)
        return np.cumsum(self._false_positives)

    @property
    def false_negatives(self):
        false_positive_arr = 1 - np.array(self._true_positives)
        return np.cumsum(false_positive_arr)

    def add_true_positives(self, value):
        self._true_positives += value

    def add_true_positives_number(self, value):
        self._true_positives_total += value

    def add_false_positives(self, value):
        self._false_positives += value

    def add_ground_truths(self, value):
        self.ground_truth_total += value

    def compare_boxes(self, boxes_ground_truth, boxes_prediction):

        matches_true_positives = []

        for elem_gt in boxes_ground_truth:
            matched = False
            for idx, elem_res in enumerate(boxes_prediction):
                iou = metricutils.bb_intersection_over_union(elem_gt.bbox, elem_res.bbox)
                if iou >= config.IOU_THRESHOLD:
                    matched = True
                    self.add_true_positives_number(1)
                    del boxes_prediction[idx]
                    break
            matches_true_positives.append(1 if matched else 0)
        self.add_true_positives(matches_true_positives)


    def __repr__(self):
        return f"Ground truth total: {self.ground_truth_total}, " \
               f"Number of images: {self.number_of_images}," \
               f"True positives: {self._true_positives_total} "
