
import sys
sys.path.append('..')

import json
import time
from utils import newresultutils

inference_result = newresultutils.InferenceRes()

if __name__ == "__main__":
    start_time = time.time()
    with open('ground_truth_boxes.json') as infile:
        gt_boxes = json.load(infile)

    with open('predicted_boxes.json') as infile:
        pred_boxes = json.load(infile)

    # Runs it for one IoU threshold
    iou_thr = 0.7

    inference_result.pred_boxes = pred_boxes
    inference_result.gt_boxes = gt_boxes
    data = inference_result.get_avg_precision_at_iou(iou_thr=iou_thr)
    end_time = time.time()
    print('Single IoU calculation took {:.4f} secs'.format(end_time - start_time))
    print('avg precision: {:.4f}'.format(data['avg_prec']))

    start_time = time.time()
    inference_result.calculate_and_graph()

    end_time = time.time()
    print('\nPlotting and calculating mAP takes {:.4f} secs'.format(end_time - start_time))
