import torch
import torch.nn as nn
import sys
import time

class YoloExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers = ["conv1", "maxpool", "layer1", "avgpool", "fc"]):
        super(YoloExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        img_size = x.size(2)
        feature_outputs, layer_outputs, yolo_outputs = [], [], []
        yolomodule = self.submodule
        for i, (module_def, module) in enumerate(zip(yolomodule.module_defs, yolomodule.module_list)):
            start_time = time.clock()
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                combined_outputs = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
                group_size = combined_outputs.shape[1] // int(module_def.get("groups", 1))
                group_id = int(module_def.get("group_id", 0))
                x = combined_outputs[:, group_size * group_id : group_size * (group_id + 1)] # Slice groupings used by yolo v4
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                x = module[0](x, img_size)
                yolo_outputs.append(x)
            layer_outputs.append(x)
            feature_outputs.append(torch.tensor([(time.clock()-start_time, x.numel()*x.element_size())]))

        return torch.cat(feature_outputs,0), torch.cat(yolo_outputs, 1)
        # return yolo_outputs if yolomodule.training else torch.cat(yolo_outputs, 1)

class ResnetFeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers = ["conv1", "maxpool", "layer1", "avgpool", "fc"]):
        super(ResnetFeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers
 
    # 自己修改forward函数
    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name == "fc": x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs, x

all_extractors = {
    'yolo': lambda x: YoloExtractor(x),
    'resnet50': lambda x: ResnetFeatureExtractor(x)
}
