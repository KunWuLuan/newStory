import torch
import torchvision.transforms as transforms
import argparse
import include.imagesreader as imagesreader
import include.videoreader as videoreader
import include.transformers as transforms
import include.extractors as extractors
import include.mymodels as mymodels

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--type', required=True, choices=['image','video'])
    parser.add_argument('--path', required=True)
    parser.add_argument('--net', default='yolo', choices=['yolo'])
    parser.add_argument('--weightpath')
    parser.add_argument('--modelpath')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    model = mymodels.all_models[args.net](args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    extractor = extractors.all_extractors[args.net](model).to(device)
    extractor.eval()
    if args.path == 'image':
        transform = transforms.all_pil_transforms[args.net]
        reader = imagesreader(args.path, transform)
    else:
        transform = transforms.all_cv_transforms[args.net]
        reader = videoreader(args.path, transform)

    next_frame = reader.next()
    last_features = None
    feature_diff = []
    while next_frame:
        if not last_features:
            feature_diff.append()
        with torch.no_grad():
            frame = next_frame.to(device)
            last_features = extractor(next_frame)
        next_frame = reader.next()
