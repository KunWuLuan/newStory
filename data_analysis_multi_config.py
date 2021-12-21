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

if __name__ == '__main__':
    args = parse()
    if not os.path.isdir(args.path):
        print('path must be dir.')
        exit(0)
    
    name = os.path.basename(args.path)
    files = os.listdir(args.path)

    fd = []
    for file in files:
        data = analysis_single_file(os.path.join(args.path, file))
        if data is None:
            continue
        fd.append(data)

    num_layers = fd[0][0].shape[1]
    num_batches = fd[0][0].shape[0]
    for i in range(num_layers):
        fig = plt.figure()
        plt.subplot(2,1,1)
        for j in range(len(fd)):
            metrics_data = fd[j][1]
            plt.plot(range(num_batches), metrics_data[:, 3].T, linestyle='--')
        plt.subplot(2,1,2)
        for j in range(len(fd)):
            layer_data = fd[j][0]
            plt.plot(range(num_batches), layer_data[:, i].T)
        plt.show()

    

    