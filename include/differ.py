from pytorchyolo.utils.utils import get_batch_statistics, compute_ap
from pytorchyolo.test import print_eval_stats
import torch
import numpy as np
from . import cococlasses

iou_thres = 0.7
verbose = False

def feature_diff(golden, o):
    similarity = torch.zeros(len(golden))
    for i in range(len(golden)):
        similarity[i] = torch.cosine_similarity(golden[i].reshape(1,-1), o[i].reshape(1,-1))
    return similarity

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

    if len(sample_metrics) == 0:  # No detections over whole validation set.
        print("---- No detections over whole validation set ----")
        return None

    if len(labels) == 0:
        return {}
    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    metrics_output = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    metrics_dict = {}
    for i, cls in enumerate(metrics_output[-1]):
        metrics_dict[cls] = torch.Tensor([metrics_output[0][i],metrics_output[1][i],metrics_output[2][i],metrics_output[3][i]])

    return metrics_dict

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.

    该函数和pytorchyolo的完全一样，只是除去了进度条
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")

