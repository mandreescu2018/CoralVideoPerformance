import numpy as np
from utils import metricutils
from config import config

class BoundingBox:
    def __init__(self, bbox):
        self.xmin = bbox.xmin
        self.ymin = bbox.ymin
        self.xmax = bbox.xmax
        self.ymax = bbox.ymax

    def __eq__(self, other):
        if metricutils.bb_intersection_over_union(self, other) >= config.IOU_THRESHOLD:
            return True
        return False


class InferenceResult:
    def __init__(self):
        self.init_object()
        self.average_precisions = []

    def init_object(self):
        self.ground_truth_total = 0
        self._true_positives = []
        self._false_negatives = []
        self._false_positives = []
        self._false_positives_total = 0
        self._false_negatives_total = 0
        self._true_positives_total = 0


    @property
    def accuracy(self):
        accuracy_val = (self._true_positives_total/self.ground_truth_total) * 100
        return round(accuracy_val, 3)

    @property
    def recall(self):

        return round(self._true_positives_total / self.ground_truth_total, 3)

    @property
    def FN_total(self):
        return sum(1 - np.array(self._true_positives))
        # return np.count_nonzero(1 - np.array(self._true_positives))

    @property
    def TP_total(self):
        return sum(self._true_positives)


    @property
    def FP_total(self):
        pass

    @property
    def precision(self):
        return round(self._true_positives_total/ (self._true_positives_total + self._false_positives_total), 3)

    @property
    def mean_average_precision(self):
        pass

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

        self.add_ground_truths(len(boxes_ground_truth))

        ground_truth_objects = [BoundingBox(item.bbox) for item in boxes_ground_truth]
        ground_truth_objects.sort(key=lambda x: x.xmin)

        prediction_objects = [BoundingBox(item.bbox) for item in boxes_prediction]
        prediction_objects.sort(key=lambda x: x.xmin)

        for gtidx, ground_truth in enumerate(ground_truth_objects):

            if len(prediction_objects) == 0:
                return
            for idx, prediction in enumerate(prediction_objects):
                if ground_truth == prediction:
                    self.add_true_positives_number(1)
                    self._true_positives.append(1)

                    del prediction_objects[idx]
                    del ground_truth_objects[gtidx]
                    break
            else:
                self._true_positives.append(0)
        self._false_negatives_total += len(ground_truth_objects)
        self._false_positives_total += len(prediction_objects)



    def __repr__(self):
        return f"Ground truth total: {self.ground_truth_total}, True positives: {self._true_positives_total}, " \
               f"Accuracy: {self.accuracy}% Recall: {self.recall} "

    def __str__(self):
        return f"Accuracy: {self.accuracy}, precision: {self.precision} , " \
               f"true_positives: {self._true_positives}\n recall: {self.recall}, \n " \
               f"number of false neg {self.FN_total},  true positives {self.TP_total}"
