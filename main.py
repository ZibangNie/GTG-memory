#!/usr/bin/python2.7

import torch
from runner import Runner
import os
import argparse
import random
import numpy as np

# seeds are freezed
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)


parser = argparse.ArgumentParser()

parser.add_argument('--dir', default='debug', type=str)
parser.add_argument('--eval', action='store_true')
parser.add_argument('--vis', action='store_true')
parser.add_argument('--config', type=str)

parser.add_argument(
    '--load_dir',
    default=None,
    type=str,
    help='Checkpoint directory name to load from during eval, e.g. best'
)
parser.add_argument(
    '--save_dir',
    default=None,
    type=str,
    help='Output directory name to save eval results to during eval, e.g. erm_v1_eval_tea_0404_run1'
)

# 新增
parser.add_argument(
    '--dump_debug',
    action='store_true',
    help='Dump per-video model / ERM debug json during evaluation'
)
parser.add_argument(
    '--debug_max_videos',
    default=-1,
    type=int,
    help='Maximum number of videos to dump debug json for; -1 means all'
)

args = parser.parse_args()

run = Runner(args)
if args.eval:
    run.evaluate()
else:
    run.train()