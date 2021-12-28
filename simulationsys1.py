import argparse
import os
import matplotlib.pyplot as plt

def parse():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_path', required=True)
    args = parser.parse_args()
    return args
    
def read_trace(file_name):
    file = open(file_name, 'r')
    if file.closed:
        return -1
    lines = file.readlines()
    batch_trace = []
    layer_trace = []
    for line in lines:
        line = line.strip()
        # print(line)
        if len(line) > 0 and line[0] == '[':
            if line[1] == '[':
                start = 2
            else:
                start = 1
            if line[-2] == ']':
                end = -2
            else:
                end = -1
            data = list(map(float,line[start:end].split(' ')))
            layer_trace.append(data)
            if line[-2] == ']':
                batch_trace.append(layer_trace)
                layer_trace = []
    file.close()
    return batch_trace

def get_accuracy_data(data_path, keys):
    file = open(data_path, 'r')
    acc = {key:[] for key in keys}
    for i,line in enumerate(file.readlines()):
        d = list(map(float, line.strip().split('\t')))
        acc[keys[i]] = sum(d)/len(d)
    file.close()
    return acc

def get_pre_processing_time(layer_trace, i):
    t = 0
    for l in range(i+1):
        t = t + layer_trace[l][0]
    return t

def get_post_processing_time(layer_trace, i):
    t = 0
    for l in range(i+1, len(layer_trace)):
        t = t + layer_trace[l][0]
    return t

def load_price(path):
    file = open(path, 'r')
    for line in file.readlines():
        s = line.split(':')
        price[int(s[0])][s[1]] = float(s[2])      
    file.close()  

price = [{},{},0.00000887345679] # price per second
bandwidth = [10, 50] # Mbps

def cal_price(c1, c2, bw, t1, t2, t3):
    return price[0][c1]*t1 + price[1][c2]*t2 + t3*price[2]*bw
    # return price[0][c1]*max(t1, 1) + price[1][c2]*max(t2, 1) + t3*price[2]*bw

if __name__ == '__main__':
    args = parse()
    load_price(os.path.join(args.data_path, 'price.txt'))
    cpu_dir = os.path.join(args.data_path, 'cpu')
    gpu_dir = os.path.join(args.data_path, 'gpu')
    num_instance_types = [len(os.listdir(cpu_dir)), len(os.listdir(gpu_dir))]
    assert(len(price[0]) == num_instance_types[0])
    assert(len(price[1]) == num_instance_types[1]+num_instance_types[0])
    trace = {}
    config_list = []
    num_layers = -1
    num_batches = -1
    for file in os.listdir(cpu_dir):
        file_path = os.path.join(cpu_dir,file)
        num_configs = len(os.listdir(file_path))
        t = {}
        if len(config_list) == 0:
            config_list = os.listdir(file_path)
        for config in os.listdir(file_path):
            t[config] = read_trace(os.path.join(file_path,config))
            if num_batches == -1:
                num_batches = len(t[config])
            if num_layers == -1:
                num_layers = len(t[config][0])
        trace['cpu/{}'.format(file)]=t

    for file in os.listdir(gpu_dir):
        file_path = os.path.join(gpu_dir,file)
        t = {}
        for config in os.listdir(file_path):
            t[config] = read_trace(os.path.join(file_path,config))
        trace['gpu/{}'.format(file)]=t

    legend = []

    acc = get_accuracy_data(os.path.join(args.data_path, 'accuracy.txt'), config_list)

    plt.figure()
    kkk = 0
    for B in bandwidth:
        # # ax = Axes3D(fig)
        # for idx,c in enumerate([config_list[3],config_list[8],config_list[10]]):
        for idx,c in enumerate([config_list[3],config_list[11],config_list[15]]):
        # for idx,c in enumerate(config_list):
            kkk = kkk+1
            redpointy, redpointx, redpointsize = 1000000000000, -1, 1000000000000
            # plt.subplot(2, 8, kkk)
            plt.subplot(2, 3, kkk)
            x,y,size,cmap = [],[],[],[]
            subidx = 0
            for i in trace.keys():
                if 'gpu' in i:
                    continue
                # if i != 'cpu/1':
                #     continue
                for j in trace.keys():
                    # if j != 'cpu/1':
                    #     continue
                    for l in range(10, 17):
                    # for l in range(num_layers):
                        total_time = 0
                        total_price = 0
                        for t in range(num_batches):
                            layer_trace1 = trace[i][c][t]
                            layer_trace2 = trace[j][c][t]
                            t1 = get_pre_processing_time(layer_trace1, l)
                            t2 = get_post_processing_time(layer_trace2, l)
                            data = layer_trace1[l][1]/(1024*128) #Bytes to Mb
                            t3 = data/B/8 # 假设有8倍的压缩率
                            total_time = total_time+t1+t2+t3
                            total_price = total_price+cal_price(i, j, B, t1, t2, t3)
                        # y.append(total_time)
                        # x.append(l)
                        # size.append(total_price*100)
                        subidx = subidx + 1
                        # if total_time < 10 and total_price < redpointsize:
                            # redpointy = total_time
                            # redpointx = subidx
                            # redpointsize = total_price
                        if total_price < 0.05 and total_time < redpointsize:
                            redpointy = total_price
                            redpointx = subidx
                            redpointsize = total_time
                        x.append(subidx)
                        y.append(total_price)
                        size.append(total_time/10)
                        if total_time < 10:
                            cmap.append('#17becf')
                        else:
                            cmap.append('#1f77b4')
                        # y.append(total_time)
                        # size.append(total_price*300)
            # plt.scatter(x, y, size, marker='o', facecolors='none', edgecolors='b')
            plt.scatter(x, y, size, c=cmap)
            # print(redpointx, redpointy, redpointsize)
            # plt.scatter(redpointx, redpointy, redpointsize*300, c='r')
            plt.scatter(redpointx, redpointy, redpointsize/10, c='r')
            # plt.legend(['all choices', 'the best choice'])
            # plt.ylim([min(y)-1, 15])
            plt.ylim([0,0.010])
            plt.xlabel('instance types')
            plt.ylabel('cost')
                    # plt.plot(x, y)
                    # legend.append('{}'.format(c))
            # print(acc[c])
            # plt.scatter(x, y)
        # plt.legend(legend)
    plt.show()