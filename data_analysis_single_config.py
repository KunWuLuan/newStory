from genericpath import isfile
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from include.util.data import analysis_single_file

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

def draw_with_data(name):
    data = analysis_single_file(name)
    if data is None:
        return
    layer_data, metrics_data = data[0], data[1]

    metrics_data = metrics_data.T
    draw_img_with_two_dim(metrics_data)

    layer_data = layer_data.T
    draw_img_with_two_dim(layer_data)

if __name__ == '__main__':
    args = parse()
    print('path is {}'.format(args.path))
    if os.path.isdir(args.path):
        name = os.path.basename(args.path)
        files = os.listdir(args.path)
        for file in files:
            analysis_single_file(os.path.join(args.path, file))
    elif os.path.isfile(args.path):
        analysis_single_file(args.path)
    else:
        exit(-1)
