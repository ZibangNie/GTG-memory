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

# original argument, mainly used for training run folder name
parser.add_argument('--dir', default='debug', type=str)

# eval / visualization flags
parser.add_argument('--eval', action='store_true')
parser.add_argument('--vis', action='store_true')

# config path
parser.add_argument('--config', type=str)

# new eval-only controls:
# load_dir = where to read checkpoint from
# save_dir = where to save outputs to
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

args = parser.parse_args()

run = Runner(args)
if args.eval:
    run.evaluate()
else:
    run.train()