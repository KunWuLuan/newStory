import torch
import torchvision.models as models
import torch.nn as nn

class YoloExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers = ["conv1", "maxpool", "layer1", "avgpool", "fc"]):
        super(YoloExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        img_size = x.size(2)
        layer_outputs, yolo_outputs = [], []
        yolomodule = self.submodule
        for i, (module_def, module) in enumerate(zip(yolomodule.module_defs, yolomodule.module_list)):
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
        return layer_outputs
        # return yolo_outputs if yolomodule.training else torch.cat(yolo_outputs, 1)
