from genericpath import isfile
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path', required=True)
    args = parser.parse_args()
    return args

def draw_img_with_two_dim(m):
    x = range(m.shape[1])
    fig = plt.figure()
    for i in range(m.shape[0]):
        plt.plot(x, m[i])
    plt.show()

def analysis_single_file(name):
    file = open(name)
    lines = file.readlines()
    is_metrics = True
    metrics_data = None
    layer_data = None
    s = ''
    for line in lines:
        s = s + line[:-1]
        if s[-1] == ']':
            # print(is_metrics, s)
            if is_metrics:
                metrics = np.array([list(map(float,s[1:-1].split()))])
                # print(metrics_data)
                if metrics_data is None:
                    metrics_data = metrics
                else:
                    metrics_data = np.concatenate((metrics_data, metrics), axis=0)
                is_metrics = not is_metrics
            else:
                layer_similarity = np.array([list(map(float,s[1:-1].split()))])
                # print(layer_data)
                if layer_data is None:
                    layer_data = layer_similarity
                else:
                    layer_data = np.concatenate((layer_data, layer_similarity), axis=0)
                is_metrics = not is_metrics
            s = ''
    metrics_data = metrics_data.T
    draw_img_with_two_dim(metrics_data)

    layer_data = layer_data.T
    draw_img_with_two_dim(layer_data)

if __name__ == '__main__':
    args = parse()
    print('path is {}'.format(args.path))
    if os.path.isdir(args.path):
        name = os.path.basename(args.path)

    elif os.path.isfile(args.path):
        analysis_single_file(args.path)
    else:
        exit(-1)
