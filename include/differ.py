from pytorchyolo.utils.utils import get_batch_statistics, ap_per_class
from pytorchyolo.test import print_eval_stats
import torch
import numpy as np
from . import cococlasses

iou_thres = 0.7
verbose = True

def feature_diff(golden, o):
    pass

def outputs2targets(golden):
    targets = torch.zeros(len(golden), 6)
    for i in range(targets.shape[0]):
        targets[i, 0] = i
        targets[i, 1] = float(golden[i, -1])
        targets[i, 2] = float(golden[i, 0])
        targets[i, 3] = float(golden[i, 1])
        targets[i, 4] = float(golden[i, 2])
        targets[i, 5] = float(golden[i, 3])
    return targets

def output_diff(golden, o):
    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    targets = outputs2targets(golden)
    labels += targets[:, 1].tolist()
    sample_metrics += get_batch_statistics([o], targets, iou_threshold=iou_thres)
    # print(targets)

    if len(sample_metrics) == 0:  # No detections over whole validation set.
        print("---- No detections over whole validation set ----")
        return None

        # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [
        np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    metrics_output = ap_per_class(
        true_positives, pred_scores, pred_labels, labels)

    print_eval_stats(metrics_output, cococlasses.classes, verbose)

    return metrics_output
