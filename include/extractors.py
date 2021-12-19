import resnetextractor
import yoloextractor
import torchvision.models as models
import pytorchyolo.models as yolomodels

all_extractors = {
    'yolo': lambda x: yoloextractor.YoloExtractor(x),
    'resnet50': lambda x: resnetextractor.ResnetFeatureExtractor(x)
}