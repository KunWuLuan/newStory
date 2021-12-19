import torchvision.models as models
import pytorchyolo.models as yolomodels

all_models = {
    'yolo': lambda x: yolomodels.load_model(x.modelpath, x.weightpath),
    'resnet50': models.resnet50(pretrained=True)
}

