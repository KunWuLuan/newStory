import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import include.resnetextractor as resnetextractor
import argparse
import include.imagesreader as imagesreader
import include.transformers as transforms
import include.extractors as extractors
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--type', required=True, choices=['image','video'])
    parser.add_argument('--path', required=True)
    parser.add_argument('--net', default='yolov3', choices=['yolov3'])
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    transform = transforms.all_transforms[args.net]
    extractor = extractors.all_extractors[args.net]()
    if args.path == 'image':
        reader = imagesreader(args.path, transform)

    next_frame = reader.next()
    last_features = None
    feature_diff = []
    while next_frame:
        if not last_features:
            last_features = extractor(next_frame)
        else:
            feature_diff.append()
        next_frame = reader.next()
