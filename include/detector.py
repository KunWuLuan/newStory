from pytorchyolo.utils.utils import rescale_boxes, non_max_suppression
import torch
import numpy as np
from . import transformers as transforms

all_detectors = {
    'yolo': lambda x,y,z: YoloDetector(x,y,z)
}

class Detector:
    def __init__(self, extractor, transform, device, size = 416) -> None:
        self.extractor = extractor
        self.transform = transform
        self.device = device
        self.size = size

class YoloDetector(Detector):
    def __init__(self, extractor, transform, device, size = 416) -> None:
        super().__init__(extractor, transform, device, size)

    def detect(self, frame, dim, to_numpy=True):
        det_frame = self.transform((frame, np.zeros((1, 5))))[0].unsqueeze(0)
        det_frame = det_frame.to(self.device)
        with torch.no_grad():
            cur_features, detections = self.extractor(det_frame)

        conf_thres = 0.5
        nms_thres = 0.5
        detections = non_max_suppression(detections, conf_thres, nms_thres)
        detections = rescale_boxes(detections[0], self.size, dim)
        if to_numpy:
            return cur_features, detections.numpy()
        else:
            return cur_features, detections

class ResnetDetector(Detector):
    def __init__(self, extractor, transform, device, size = 416) -> None:
        super().__init__(extractor, transform, device, size)

    def detect(self, frame):
        det_frame = self.transform(frame)
        det_frame = det_frame.to(self.device)
        with torch.no_grad():
            cur_features, detections = self.extractor(det_frame)
        return cur_features, detections