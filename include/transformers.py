import torchvision.transforms as transforms
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS

img_size=416

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

all_cv_transforms = {
    'yolov3':transforms.Compose([DEFAULT_TRANSFORMS,
        Resize(img_size)]),
    'resnet':transforms.Compose([
        DEFAULT_TRANSFORMS,
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ]),
}

all_pil_transforms = {
    'yolov3':transforms.Compose([
        Resize(img_size)]),
    'resnet':transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ]),
}