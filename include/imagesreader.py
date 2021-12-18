import os
from PIL import Image

class ImagesReader:
    def __init__(self, path, suffix, transform):
        self.path = path
        self.transform = transform
        self.dataset = []
        self.cursor = 0
        files = os.listdir(path)
        for file in files: #遍历文件夹
            if not os.path.isdir(file) and suffix in file: #判断是否是文件夹，不是文件夹才打开
                self.data.append(self.transform(Image.open(os.path.join(path, file))))

    def next(self):
        if self.cursor == len(self.dataset):
            return None
        else:
            ret = self.dataset[self.cursor]
            self.cursor = self.cursor-1
            return ret
