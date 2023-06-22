import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from .configs.default import get_cfg
from .configs.things_eval import get_cfg as get_things_cfg
from .configs.small_things_eval import get_cfg as get_small_things_cfg
from .core.utils.misc import process_cfg
import datasets
from .core.utils import flow_viz
from .core.utils import frame_utils

# from FlowFormer import FlowFormer
from .core.FlowFormer import build_flowformer
from .core.raft import RAFT

from .core.utils.utils import InputPadder, forward_interpolate

import argparse
def get_matcher(path,small):
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model', help="restore checkpoint")
    #parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision',default="True")
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args("".split())
    cfg = get_cfg()
    if args.small:
        cfg = get_small_things_cfg()
    else:
        cfg = get_things_cfg()
    cfg.update(vars(args))

    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(path))
    #model = model.module
    print(args)

    model.cuda()
    model.eval()
    return model
