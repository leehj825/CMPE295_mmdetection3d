# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet3d.utils import replace_ceph_backend

import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()

    return args

def average_weights(client_weights_list):
    num_clients = len(client_weights_list)
    avg_dict = {}

    for k in client_weights_list[0].keys():
        weights_sum = sum([state_dict[k] for state_dict in client_weights_list])
        avg_dict[k] = weights_sum / num_clients
    return avg_dict

def write_model_weights_to_file(weights, filename):
    with open(filename, 'w') as file:
        for key, value in weights.items():
            file.write(f'{key}: {value}\n')


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)

    cfg.launcher = args.launcher

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    map_location = 'cuda:0'

    # load checkpoints
    ckpt1 = torch.load('work_dirs/pgd_r101-caffe_fpn_head-gn_4xb3-4x_waymoD5-fov-mono3d_n1/epoch_5.pth', map_location=map_location)
    ckpt2 = torch.load('work_dirs/pgd_r101-caffe_fpn_head-gn_4xb3-4x_waymoD5-fov-mono3d_n2/epoch_5.pth', map_location=map_location)
    ckpt3 = torch.load('work_dirs/pgd_r101-caffe_fpn_head-gn_4xb3-4x_waymoD5-fov-mono3d_n3/epoch_5.pth', map_location=map_location)
    ckpt4 = torch.load('work_dirs/pgd_r101-caffe_fpn_head-gn_4xb3-4x_waymoD5-fov-mono3d_n4/epoch_5.pth', map_location=map_location)
    ckpt5 = torch.load('work_dirs/pgd_r101-caffe_fpn_head-gn_4xb3-4x_waymoD5-fov-mono3d_n5/epoch_5.pth', map_location=map_location)
    ckpt6 = torch.load('work_dirs/pgd_r101-caffe_fpn_head-gn_4xb3-4x_waymoD5-fov-mono3d_n6/epoch_5.pth', map_location=map_location)
    ckpt7 = torch.load('work_dirs/pgd_r101-caffe_fpn_head-gn_4xb3-4x_waymoD5-fov-mono3d_n7/epoch_5.pth', map_location=map_location)
    ckpt8 = torch.load('work_dirs/pgd_r101-caffe_fpn_head-gn_4xb3-4x_waymoD5-fov-mono3d_n8/epoch_5.pth', map_location=map_location)
    # ckpt = torch.load(args.checkpoint, map_location=map_location)

    # aggregate checkpoints weights
    client_weights_list = [ckpt1['state_dict'], ckpt2['state_dict'], ckpt3['state_dict'], ckpt4['state_dict'], ckpt5['state_dict'], ckpt6['state_dict'], ckpt7['state_dict'], ckpt8['state_dict']]
    global_weights = average_weights(client_weights_list)

    write_model_weights_to_file(ckpt1['state_dict'], 'checkpoint_1_weights.txt')
    write_model_weights_to_file(ckpt2['state_dict'], 'checkpoint_2_weights.txt')
    write_model_weights_to_file(ckpt3['state_dict'], 'checkpoint_3_weights.txt')
    write_model_weights_to_file(ckpt4['state_dict'], 'checkpoint_4_weights.txt')
    write_model_weights_to_file(ckpt4['state_dict'], 'checkpoint_5_weights.txt')
    write_model_weights_to_file(ckpt4['state_dict'], 'checkpoint_6_weights.txt')
    write_model_weights_to_file(ckpt4['state_dict'], 'checkpoint_7_weights.txt')
    write_model_weights_to_file(ckpt4['state_dict'], 'checkpoint_8_weights.txt')
    write_model_weights_to_file(global_weights, 'global_weights.txt')

    global_model = torch.load('work_dirs/pgd_r101-caffe_fpn_head-gn_4xb3-4x_waymoD5-fov-mono3d_n1/epoch_5.pth')

    # keys
    #   - meta
    #   - state_dict
    #   - message_hub
    #   - optimizer
    #   - param_schedulers

    # edit state_dict of old model and save it
    global_model['state_dict'] = global_weights
    torch.save(global_model, 'global_weight_pgd.pth')



if __name__ == '__main__':
    main()