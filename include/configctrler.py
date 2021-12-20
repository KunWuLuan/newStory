from . import transformers
from . import detector
import cv2 as cv

all_configs = [
    [1, (1280,720)], 
    # [1, (854,480)], [2, (1280,720)], 
    [2, (854,480)],
]

def build_controllers(net, extractor, device):
    controllers = []
    for config in all_configs:
        controllers.append(Controller(net, extractor, device, config[1], config[0]))
    return controllers

class Controller:
    def __init__(self, net, extractor, device, img_size, freq) -> None:
        self.img_size = img_size
        self.freq = freq
        self.extractor = extractor
        self.idx = 0
        self.last_features = None
        self.last_outputs = None
        self.transform = transformers.all_cv_transforms[net](416)
        self.detector = detector.all_detectors[net](self.extractor, self.transform, device)

    def get_freq(self):
        return self.freq

    def execute_once(self, frame):
        resized_frame = cv.resize(frame, self.img_size)
        if self.idx == 0:
            self.last_features, self.last_outputs = self.detector.detect(resized_frame, frame.shape[:2], False)
        self.idx = (self.idx + 1)%self.freq
        return self.last_features, self.last_outputs
