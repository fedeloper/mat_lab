from __future__ import print_function, division
import sys
# sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import rasterio
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomRotation, RandomResizedCrop, ToTensor

from FlowFormer.core.utils.flow_transforms import Compose

import evaluate_FlowFormer as evaluate


from core.loss import sequence_loss
from core.optimizer import fetch_optimizer
from core.utils.misc import process_cfg
from loguru import logger as loguru_logger

# from torch.utils.tensorboard import SummaryWriter
from core.utils.logger import Logger

# from core.FlowFormer import FlowFormer
from core.FlowFormer import build_flowformer


def get_list():
    path_of_the_directory = './data/dataset/'
    paths = []
    for filename in os.listdir(path_of_the_directory):
        if filename[0] != ".":
            f = os.path.join(path_of_the_directory, filename)
            if len(os.listdir(f)) > 3:
                if not os.path.isfile(f):
                    lst = sorted([os.path.abspath(os.path.join(f, p)) for p in os.listdir(f)])
                    lst = [item for item in lst if "az_rg" not in item and "._" not in item]
                    if len(lst) == 4:
                        paths.append(lst)
    return paths


def normalize(band):
    band_min, band_max = (np.nanmin(band), np.nanmax(band))
    return ((band - band_min) / ((band_max - band_min)))


def gammacorr(band):
    gamma = 2
    return np.power(band, 1 / gamma)
def normalize_sar(band, percentile):
  p = np.nanpercentile(band,percentile)
  img = band.clip(0,p)
  img_n = (img-np.nanmin(img))/(np.nanmax(img)-np.nanmin(img))
  img_n[img_n==np.nan] = 0
  return img_n

def create_RGB_composite(red,green,blue):

  red_g=gammacorr(red)
  blue_g=gammacorr(blue)
  green_g=gammacorr(green)

  red_gn = normalize(red_g)
  green_gn = normalize(green_g)
  blue_gn = normalize(blue_g)

  rgb_composite= np.dstack((red_gn, green_gn, blue_gn))
  return rgb_composite
class UAVSARDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        self.sample_paths = sorted(list(get_list()))
        self.transform = transform

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        #sample_dir = self.sample_paths[idx]
        path_s2, sentinel_path, llh_path, uavsar_path= self.sample_paths[idx]
        # Load uavsar.jpg

        uavsar_img = Image.open(uavsar_path)

        # Load uavsar.llh
        llh = np.load(llh_path)

        # Load sentinel1.tif
        s1_img = normalize_sar(rasterio.open(sentinel_path).read()[0, :, :], 97)
        sentinel2= rasterio.open(sentinel_path).read()
        s2_norm_rgb = create_RGB_composite(sentinel2[3, :, :], sentinel2[2, :, :], sentinel2[1, :, :])



        if self.transform:
            uavsar_img, llh = self.transform(uavsar_img, llh)

        return uavsar_img, llh,s1_img,rasterio.open(sentinel_path)

# Your function to fetch dataloader
def fetch_dataloader(cfg):
    transform = Compose([RandomRotation(30), RandomResizedCrop(224), ToTensor()])
    dataset = UAVSARDataset("../Tester/data/dataset", transform=transform)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    return dataloader

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

#torch.autograd.set_detect_anomaly(True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(cfg):
    model = nn.DataParallel(build_flowformer(cfg))
    loguru_logger.info("Parameter Count: %d" % count_parameters(model))

    if cfg.restore_ckpt is not None:
        print("[Loading ckpt from {}]".format(cfg.restore_ckpt))
        model.load_state_dict(torch.load(cfg.restore_ckpt), strict=True)

    model.cuda()
    model.train()

    train_loader = fetch_dataloader(cfg)
    optimizer, scheduler = fetch_optimizer(model, cfg.trainer)

    total_steps = 0
    scaler = GradScaler(enabled=cfg.mixed_precision)
    logger = Logger(model, scheduler, cfg)

    add_noise = False

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            uavsar_img, llh, s1_img, data_sentinel = data_blob

            if cfg.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                uavsar_img = (uavsar_img + stdv * torch.randn(*uavsar_img.shape).cuda()).clamp(0.0, 255.0)
                s1_img = (s1_img + stdv * torch.randn(*s1_img .shape).cuda()).clamp(0.0, 255.0)
            print(uavsar_img.shape,s1_img.shape)
            uavsar_img, s1_img = torch.tensor( uavsar_img).cuda(), torch.tensor(s1_img).cuda()
            output = {}
            flow_predictions = model(uavsar_img , s1_img, output)
            print(flow_predictions.shape)
            loss, metrics = sequence_loss(flow_predictions, None, None, cfg)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            metrics.update(output)
            logger.push(metrics)

            ### change evaluate to functions

            if total_steps % cfg.val_freq == cfg.val_freq - 1:
                PATH = '%s/%d_%s.pth' % (cfg.log_dir, total_steps+1, cfg.name)
                # torch.save(model.state_dict(), PATH)

                results = {}
                for val_dataset in cfg.validation:
                    if val_dataset == 'chairs':
                        results.update(evaluate.validate_chairs(model.module))
                    elif val_dataset == 'sintel':
                        results.update(evaluate.validate_sintel(model.module))
                    elif val_dataset == 'kitti':
                        results.update(evaluate.validate_kitti(model.module))

                logger.write_dict(results)
                
                model.train()
            
            total_steps += 1

            if total_steps > cfg.trainer.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = cfg.log_dir + '/final'
    torch.save(model.state_dict(), PATH)

    PATH = f'checkpoints/{cfg.stage}.pth'
    torch.save(model.state_dict(), PATH)

    return PATH

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='flowformer', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    args = parser.parse_args()

    #if args.stage == 'chairs':
    from configs.default import get_cfg
    #elif args.stage == 'things':
    #    from configs.things import get_cfg
    #elif args.stage == 'sintel':
    #    from configs.sintel import get_cfg
    #elif args.stage == 'kitti':
    #    from configs.kitti import get_cfg
    #elif args.stage == 'autoflow':
    #    from configs.autoflow import get_cfg

    cfg = get_cfg()
    cfg.update(vars(args))
    process_cfg(cfg)
    loguru_logger.add(str(Path(cfg.log_dir) / 'log.txt'), encoding="utf8")
    loguru_logger.info(cfg)

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(cfg)
