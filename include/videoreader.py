import cv2 as cv

class VideoReader:
    def __init__(self, path) -> None:
        cap = cv.VideoCapture(path)
        if not cap.isOpened():
            print('Error: can not open the video path')
            raise Exception
        self.path = path
        self.cap = cap
        self.frames_count = 0

    def next(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return -1, None
        # cv.imshow('frame',frame)
        # if cv.waitKey(1) & 0xFF == ord('q'):
        #     return
        # Convert OpenCV bgr to rgb
        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # frame = transforms.ToPILImage()(frame)
        return 1, img

