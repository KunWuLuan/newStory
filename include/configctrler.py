from . import transformers
from . import detector
import cv2 as cv
import time

# all_configs = [
#     [1, (1280,720)], 
#     # [1, (854,480)], [2, (1280,720)], 
#     [2, (854,480)],
#     [4, (640,360)],
#     [4, (427,240)],
# ]

all_configs = [
    [1, 608], 
    # [1, (854,480)], [2, (1280,720)], 
    [2, 608],
    [4, 608],
    [4, 416],
]

def build_controllers(net, extractor, device):
    controllers = []
    for config in all_configs:
        controllers.append(Controller(net, extractor, device, config[1], config[0]))
    return controllers

def get_cost_of_all_constrollers(controllers):
    cost = {}
    for controller in controllers:
        cost[controller.name] = controller.cost_time
    return cost
class Controller:
    def __init__(self, net, extractor, device, img_size, freq) -> None:
        self.img_size = img_size
        self.freq = freq
        self.extractor = extractor
        self.idx = 0
        self.last_features = None
        self.last_outputs = None
        self.transform = transformers.all_cv_transforms[net](img_size)
        self.detector = detector.all_detectors[net](self.extractor, self.transform, device, img_size)
        self.cost_time = 0
        self.name = (freq, img_size)

    def get_freq(self):
        return self.freq

    def execute_once(self, frame):
        start = time.time()
        # resized_frame = cv.resize(frame, self.img_size)
        if self.idx == 0:
            self.last_features, self.last_outputs = self.detector.detect(frame, frame.shape[:2], False)
        self.idx = (self.idx + 1)%self.freq
        self.cost_time = self.cost_time + time.time()-start
        return self.last_features, self.last_outputs

    def get_cost(self):
        return self.cost_time
