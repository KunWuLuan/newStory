import torch
import torchvision.models as models
import torch.nn as nn

class ResnetFeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers = ["conv1", "maxpool", "layer1", "avgpool", "fc"]):
        super(ResnetFeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers
 
    # 自己修改forward函数
    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "fc": x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs