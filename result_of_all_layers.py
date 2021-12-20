import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import include.extractors as resnetextractor
from PIL import Image

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]

device = torch.device("cuda:0")
net = models.resnet50(pretrained=True)
net = resnetextractor.ResnetFeatureExtractor(net)
net.to(device)
net.eval()

dataset = []
img = Image.open("dog.jpg")
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)
dataset.append(batch_t)

# 使用torch.no_grad的话在前向传播中不记录梯度，节省内存
with torch.no_grad():
    for data in dataset:
        # 预测
        data = data.to(device)
        outputs = net(data)
        _, indices = torch.sort(outputs[-1], descending=True)
        percentage = torch.nn.functional.softmax(outputs[-1], dim=1)[0] * 100
        print(outputs)
        print([(classes[idx], percentage[idx].item()) for idx in indices[0][:5]])
