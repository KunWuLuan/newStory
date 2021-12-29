import argparse
import numpy as np
import torch
import include.mymodels as mymodels
import include.extractors as extractors
from include.util.expdatawriter import DataWriter
import os
import time
from include.configctrler import build_controllers, get_cost_of_all_constrollers
from include.videoreader import VideoReader
from include.differ import output_diff, feature_diff

writer = None

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path', required=True)
    parser.add_argument('--outputpath', required=True)
    parser.add_argument('--net', default='yolo', choices=['yolo'])
    parser.add_argument('--max_batch', default=-1, type=int)
    parser.add_argument('--weightpath')
    parser.add_argument('--modelpath')
    args = parser.parse_args()
    return args

def single_frame_profile(reader, controllers):
    batch_profile(reader, controllers, batch_size=1)

def batch_profile(reader, controllers, batch_size, max_batch = -1):
    print('profile start...')
    start_time = time.time()
    batch_count = 0
    ret, next_frame = reader.next()
    while ret == 1 and (max_batch == -1 or batch_count < max_batch):
        count = 0
        batch_start_time = time.time()
        # batch_feature_metrics = [None for i in range(len(controllers)-1)]
        batch_feature_metrics = [None for i in range(len(controllers))]
        batch_output_metrics = [{} for i in range(len(controllers)-1)]
        print('batch {} start...'.format(batch_count))
        while ret == 1 and count < batch_size:
            golden_feature = None
            golden_output = None
            for i, c in enumerate(controllers):
                cur_feature, cur_output = c.execute_once(next_frame)
                # print(cur_feature)
                if batch_feature_metrics[i] is None:
                    batch_feature_metrics[i] = cur_feature.unsqueeze(0)
                else:
                    batch_feature_metrics[i] = torch.cat((batch_feature_metrics[i], cur_feature.unsqueeze(0)), dim=0)
                # if golden_feature is None:
                #     golden_feature = cur_feature
                #     golden_output = cur_output
                # else:
                    # feature_output = feature_diff(golden_feature, cur_feature)
                    # if batch_feature_metrics[i-1] is None:
                    #     batch_feature_metrics[i-1] = feature_output
                    # else:
                    #     batch_feature_metrics[i-1] += feature_output

                    # metrics_output = output_diff(golden_output, cur_output)
                    # for k,v in metrics_output.items():
                    #     v = v.unsqueeze(0)
                    #     if k in batch_output_metrics[i-1].keys():
                    #         batch_output_metrics[i-1][k] = torch.cat((batch_output_metrics[i-1][k],v), dim=0)
                    #     else:
                    #         batch_output_metrics[i-1][k] = v
            count = count+1
            ret, next_frame = reader.next()
        if count == batch_size:
            writer.write(0, 'batch {} end. cost {}s ...'.format(batch_count, (time.time()-batch_start_time)))
            # print('batch {} end. cost {}s ...'.format(batch_count, (time.time()-batch_start_time)))
            # for i in range(len(batch_output_metrics)):
                # print('configuration {}'.format(i))
                # avg_metrics = None
                # for k,v in batch_output_metrics[i].items():
                #     batch_output_metrics[i][k] = torch.mean(v, dim=0)
                #     # print('class {}, mean:{}'.format(k, batch_output_metrics[i][k]))
                #     if avg_metrics == None:
                #         avg_metrics = batch_output_metrics[i][k].unsqueeze(0)
                #     else:
                #         avg_metrics = torch.cat((avg_metrics, batch_output_metrics[i][k].unsqueeze(0)), dim=0)
                # if avg_metrics is None:
                #     avg_metrics = torch.tensor([0,0,0,0])
                # else:
                #     avg_metrics = torch.mean(avg_metrics, dim=0)
                # print('avg metrics mean:{}'.format(avg_metrics))
                # writer.write(i+1, avg_metrics.numpy())

                # batch_feature_metrics[i] = batch_feature_metrics[i].div(batch_size)
                # print('tensor of different layers: {}\n'.format(batch_feature_metrics[i]))
            for i in range(len(batch_feature_metrics)):
                # print(batch_feature_metrics[i])
                # batch_feature_metrics[i] = torch.tensor(batch_feature_metrics[i])
                # print(batch_feature_metrics[i])
                s = torch.sum(batch_feature_metrics[i], dim=0).numpy()
                writer.write(i, s)
                print(i, s)
        else:
            break
        batch_count = batch_count + 1
    writer.write(0, 'end.\n cost time: {}s'.format((time.time()-start_time)))
    writer.write(0, 'cost of all controllers:\n{}'.format(get_cost_of_all_constrollers(controllers)))
    print('end.\n cost time: {}s'.format((time.time()-start_time)))


if __name__ == '__main__':
    print('system start...')
    args = parse_args()
    print('args parse completed...')
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    model = mymodels.all_models[args.net](args)
    extractor = extractors.all_extractors[args.net](model).to(device)
    extractor.eval()
    print('extractor construct completed...')

    controllers = build_controllers(args.net, extractor, device)
    writer = DataWriter(args.outputpath, os.path.basename(args.path).split('.')[0], len(controllers))
    reader = VideoReader(args.path)
    batch_profile(reader=reader, controllers=controllers, batch_size=30, max_batch=args.max_batch)
