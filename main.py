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

args = parser.parse_args()

run = Runner(args)
if args.eval:
    run.evaluate()
else:
    run.train()