import cv2 as cv
import torchvision.transforms as transforms

class VideoReader:
    def __init__(self, path, transform) -> None:
        cap = cv.VideoCapture(path)
        if not cap.isOpened():
            print('Error: can not open the video path')
            return
        self.transform = transform
        self.path = path
        self.cap = cap
        self.frames_count = 0

    def next(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return -1, None
        # Convert OpenCV bgr to rgb
        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # frame = transforms.ToPILImage()(frame)
        return 1, self.transform(frame)

