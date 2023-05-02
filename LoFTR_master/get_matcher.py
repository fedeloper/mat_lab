
from scipy.ndimage import uniform_filter, variance
from skimage.transform import AffineTransform


import PIL
from PIL import Image, ImageFilter

import rasterio as rio
from rasterio import warp

import matplotlib

from matplotlib import pyplot as plt

import argparse
import torch
from .src.loftr import LoFTR, default_cfg

def get_matcher():


    matcher = LoFTR(config=default_cfg)

    matcher.load_state_dict(torch.load("../LoFTR_master/weights/outdoor_ds.ckpt")['state_dict'])
    return matcher.eval().cuda()