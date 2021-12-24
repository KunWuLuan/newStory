import torch
import time
import torchvision.transforms as transforms
import argparse
import include.imagesreader as imagesreader
import include.videoreader as videoreader
import include.transformers as transforms
import include.extractors as extractors
import include.mymodels as mymodels
import include.detector as detectors
from pytorchyolo.utils.utils import load_classes, rescale_boxes, non_max_suppression, print_environment_info

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--type', required=True, choices=['image','video'])
    parser.add_argument('--path', required=True)
    parser.add_argument('--net', default='yolo', choices=['yolo'])
    parser.add_argument('--weightpath')
    parser.add_argument('--modelpath')
    args = parser.parse_args()
    return args

def draw_and_save_img(img, det, img_size, output_path, classes):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.ticker import NullLocator
    import random
    import numpy as np
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    # Rescale boxes to original image
    unique_labels = det[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, n_cls_preds)]
    bbox_colors = random.sample(colors, n_cls_preds)
    for x1, y1, x2, y2, conf, cls_pred in det:

        print(f"\t+ Label: {classes[int(cls_pred)]} | Confidence: {conf.item():0.4f}")

        box_w = x2 - x1
        box_h = y2 - y1

        color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
        # Create a Rectangle patch
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
        # Add the bbox to the plot
        ax.add_patch(bbox)
        # Add label
        plt.text(
            x1,
            y1,
            s=classes[int(cls_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": color, "pad": 0})

    # Save generated image with detections
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
    plt.close()


if __name__ == '__main__':
    print('system start...')
    args = parse_args()
    print('args parse completed...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = mymodels.all_models[args.net](args)
    extractor = extractors.all_extractors[args.net](model).to(device)
    extractor.eval()
    print('extractor construct completed...')

    if args.path == 'image':
        transform = transforms.all_pil_transforms[args.net](608)
        reader = imagesreader.ImagesReader(args.path)
    else:
        transform = transforms.all_cv_transforms[args.net](224)
        reader = videoreader.VideoReader(args.path)

    detector = detectors.all_detectors[args.net](extractor, transform, device)
    print('detector construct completed...')

    ret, next_frame = reader.next()
    max_count = 1
    count = 0
    last_features = None
    feature_diff = []
    print('detect start...')
    start_time = time.time_ns()
    while ret == 1 and count < max_count:
        cur_features, detections = detector.detect(next_frame, next_frame.shape[:2])
        show_res = True
        if show_res:
            print(detections)
            dets = torch.tensor(detections)
            from include.cococlasses import classes
            draw_and_save_img(next_frame, dets, transforms.get_img_size(), './det.jpg', classes)
        if last_features:
            feature_diff.append(count)
        last_features = cur_features
        ret, next_frame = reader.next()
        count = count + 1
    print('end.\n cost time: {}s'.format((time.time_ns()-start_time)/1000000000))
