import argparse
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
    ret, next_frame = reader.next()
    print('detect start...')
    start_time = time.time_ns()
    count = 0
    max_count = 10
    while ret == 1 and (max_count == -1 or count < max_count):
        golden_feature = None
        golden_output = None
        cur_features = []
        cur_outputs = []
        for c in controllers:
            cur_feature, cur_output = c.execute_once(next_frame)
            # print(cur_output)
            if golden_feature is None:
                golden_feature = cur_feature
                golden_output = cur_output
            else:
                cur_features.append(cur_feature)
                cur_outputs.append(cur_output)
                metrics = output_diff(golden_output, cur_output)
        count = count+1
        ret, next_frame = reader.next()
    print('end.\n cost time: {}s'.format((time.time_ns()-start_time)/1000000000))
