import argparse
import numpy as np
import torch
import include.mymodels as mymodels
import include.extractors as extractors
import time
from include.configctrler import build_controllers
from include.videoreader import VideoReader
from include.differ import output_diff, feature_diff

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path', required=True)
    parser.add_argument('--net', default='yolo', choices=['yolo'])
    parser.add_argument('--weightpath')
    parser.add_argument('--modelpath')
    args = parser.parse_args()
    return args

def single_frame_profile(reader, controllers):
    batch_profile(reader, controllers, batch_size=1)

def batch_profile(reader, controllers, batch_size, max_batch = -1):
    print('profile start...')
    start_time = time.time_ns()
    batch_count = 0
    ret, next_frame = reader.next()
    while ret == 1 and (max_batch == -1 or batch_count < max_batch):
        count = 0
        batch_start_time = time.time_ns()
        batch_feature_metrics = [None for i in range(len(controllers)-1)]
        batch_output_metrics = [{} for i in range(len(controllers)-1)]
        print('batch {} start...'.format(batch_count))
        while count < batch_size:
            golden_feature = None
            golden_output = None
            for i, c in enumerate(controllers):
                cur_feature, cur_output = c.execute_once(next_frame)
                if golden_feature is None:
                    golden_feature = cur_feature
                    golden_output = cur_output
                else:
                    feature_output = feature_diff(golden_feature, cur_feature)
                    if batch_feature_metrics[i-1] is None:
                        batch_feature_metrics[i-1] = feature_output
                    else:
                        batch_feature_metrics[i-1] += feature_output

                    metrics_output = output_diff(golden_output, cur_output)
                    for k,v in metrics_output.items():
                        v = v.unsqueeze(0)
                        if k in batch_output_metrics[i-1].keys():
                            batch_output_metrics[i-1][k] = torch.cat((batch_output_metrics[i-1][k],v), dim=0)
                        else:
                            batch_output_metrics[i-1][k] = v
            count = count+1
            ret, next_frame = reader.next()
        print('batch {} end. cost {}s ...'.format(batch_count, (time.time_ns()-batch_start_time)/1000000000))
        for i in range(len(batch_output_metrics)):
            for k,v in batch_output_metrics[i].items():
                print('class {}, mean:{}'.format(k, torch.mean(batch_output_metrics[i][k], dim=0)))
            print('tensor of different layers: {}'.format(batch_feature_metrics[i].div(batch_size)))
        batch_count = batch_count + 1
    print('end.\n cost time: {}s'.format((time.time_ns()-start_time)/1000000000))


if __name__ == '__main__':
    print('system start...')
    args = parse_args()
    print('args parse completed...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = mymodels.all_models[args.net](args)
    extractor = extractors.all_extractors[args.net](model).to(device)
    extractor.eval()
    print('extractor construct completed...')

    controllers = build_controllers(args.net, extractor, device)
    reader = VideoReader(args.path)
    batch_profile(reader=reader, controllers=controllers, batch_size=30, max_batch=2)