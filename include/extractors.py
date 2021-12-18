import resnetextractor
import torchvision.models as models

all_extractors = {
    'resnet50': lambda : return resnetextractor.ResnetFeatureExtractor(models.resnet50(pretrained=True))
}