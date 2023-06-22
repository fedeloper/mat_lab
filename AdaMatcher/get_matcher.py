
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

from AdaMatcher.src.config.default import get_cfg_defaults
from AdaMatcher.src.lightning.lightning_adamatcher import PL_AdaMatcher
from AdaMatcher.src.utils.profiler import build_profiler


def get_matcher(path="../AdaMatcher/weights/adamatcher.ckpt"):
    config = get_cfg_defaults()
    profiler = build_profiler("inference")
    matcher = PL_AdaMatcher(
        config,
        pretrained_ckpt=path,
        profiler=profiler,
        dump_dir=".",
    )

    return matcher.eval().cuda()