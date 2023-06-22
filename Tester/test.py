
from typing import Tuple

import cv2
import numpy as np

import PIL
from PIL import Image

from scipy.ndimage import uniform_filter, variance

from torch import onnx, Tensor
from torch.nn.utils import prune

from rasterio.crs import CRS
from rasterio import transform as tform
import torch.onnx.utils
import rasterio as rio
from rasterio import warp
import matplotlib
from matplotlib import pyplot as plt

import argparse
import pytorch_lightning as pl

# Need thsxdfsdfis to plot in HD
# This takes up a lot of memory!
matplotlib.rcParams['figure.dpi'] = 100

from tester import Tester

class Tester_Pytorch_TopicFM(Tester):
        def __init__(self,matcher,device="cuda"):
            super().__init__(matcher)
            self.name_model= "TopicFM"
            self.matcher = matcher.to(device)#get_matcher()
            self.device=device
        def make_inference(self,image0:np.numarray,image1:np.numarray)->Tuple[np.numarray,np.numarray,np.numarray]:
            #normalized_img0 = cv2.equalizeHist(image0)
            #normalized_img1 = cv2.equalizeHist(image1)
            m1, m2, c, m_bits = self.matcher.matcher(torch.from_numpy(image0).to(self.device),
                                                     torch.from_numpy(image1).to(self.device))
            return m1.cpu().numpy(),m2.cpu().numpy(),c.cpu().numpy()
class Tester_ONNX(Tester):
        def __init__(self,matcher):
            super().__init__()
            self.name_model= "TopicFM_ONNX"
            self.matcher = matcher#get_matcher()

        def make_inference(self,image0:np.numarray,image1:np.numarray)->Tuple[np.numarray,np.numarray,np.numarray]:
            mkpts0 ,mkpts1,mconf = self.matcher.run(None, {"image0": image0,"image1": image1})
            return mkpts0 ,mkpts1,mconf
class Tester_Pytorch_MatchFormer(Tester):
        def __init__(self,matcher,device):
            super().__init__(matcher)
            self.name_model= "MarchFormer"
            self.matcher = matcher.to(device)#get_matcher()
            self.device=device
        def make_inference(self,image0:np.numarray,image1:np.numarray)->Tuple[np.numarray,np.numarray,np.numarray]:

            batch = {'image0': torch.from_numpy(image0).to(self.device), 'image1': torch.from_numpy(image1).to(self.device)}

            self.matcher.matcher(batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()

            return mkpts0 ,mkpts1,mconf
class Tester_PyTorch_ASpanFormer(Tester):
        def __init__(self,matcher,device):
            super().__init__()
            self.name_model= "ASpanFormer"
            self.matcher = matcher.to(device)#get_matcher()
            self.device=device
        def make_inference(self,image0:np.numarray,image1:np.numarray)->Tuple[np.numarray,np.numarray,np.numarray]:

            batch = {'image0': torch.from_numpy(image0).to(self.device), 'image1': torch.from_numpy(image1).to(self.device)}

            self.matcher.matcher(batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()

            return mkpts0 ,mkpts1,mconf

class Tester_Pytorch_LoFTR(Tester):
        def __init__(self,matcher,device):
            super().__init__()
            self.name_model= "LoFTR"
            self.matcher = matcher.to(device)#get_matcher()
            self.device=device
        def make_inference(self,image0:np.numarray,image1:np.numarray)->Tuple[np.numarray,np.numarray,np.numarray]:
            batch = {'image0': torch.from_numpy(image0).to(self.device), 'image1': torch.from_numpy(image1).to(self.device)}
            self.matcher(batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()
            return mkpts0 ,mkpts1,mconf


import random
import numpy as np

def add_channel(arr):
    """
    Adds a new channel of zeros to a numpy array of shape (channels,h,w).

    Parameters:
    arr (numpy.ndarray): The input array with shape (channels,h,w).

    Returns:
    numpy.ndarray: The output array with shape (channels+1,h,w).
    """
    # Get the number of channels, height, and width of the input array
    channels, h, w = arr.shape

    # Create a new numpy array with an extra channel of zeros
    new_arr = np.zeros((channels+1, h, w))+0.5

    # Copy the input array into the new array, leaving the new channel as zeros
    new_arr[:-1, :, :] = arr

    return new_arr
from FlowFormer.core.utils import flow_viz
class Tester_Pytorch_FlowFormer(Tester):
        def __init__(self,matcher,device):
            super().__init__(matcher)
            self.name_model= "FlowFormer"

            self.matcher = matcher.to(device)#get_matcher()
            self.device=device
        def make_inference(self, image0: np.ndarray, image1: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            img0 = torch.as_tensor(image0, device=self.device).repeat(1, 3, 1, 1)
            img1 = torch.as_tensor(image1, device=self.device).repeat(1, 3, 1, 1)
            flow_pre, _ = self.matcher(img0, img1)

            flow_pre_cpu = flow_pre.cpu().squeeze().permute(1, 2, 0).numpy()
            flow_img = flow_viz.flow_to_image(flow_pre_cpu)
            plt.imshow(flow_img)
            plt.show()

            x_coords = np.tile(np.arange(flow_pre.shape[2]), (flow_pre.shape[3], 1)).T
            y_coords = np.tile(np.arange(flow_pre.shape[3]), (flow_pre.shape[2], 1))

            # Add the flow to the x and y coordinates
            x_flow = flow_pre[:, 0, :, :].cpu().numpy()
            y_flow = flow_pre[:, 1, :, :].cpu().numpy()
            new_x_coords = np.add(x_coords ,x_flow)
            new_y_coords = np.add(y_coords, y_flow)

            # Create the list of tuples
            mkpts0 = np.array(list(zip(x_coords.flatten(), y_coords.flatten())))
            mkpts1 = np.array(list(zip(new_x_coords.flatten(), new_y_coords.flatten())))
            mconf = np.full((len(mkpts0),), 0.98)
            a = np.zeros(mkpts0.shape[0], dtype=int)
            a[:3000] = 1
            np.random.shuffle(a)
            a = a.astype(bool)

            return mkpts0[a], mkpts1[a], mconf[a]
        def make_predicition_for_training(self,image0: Tensor, image1: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

            img0 = torch.as_tensor(image0, device=self.device).repeat(1, 3, 1, 1)
            img1 = torch.as_tensor(image1, device=self.device).repeat(1, 3, 1, 1)
            flow_pre, _ = self.matcher(img0, img1)

            flow_pre_cpu = flow_pre.cpu().squeeze().permute(1, 2, 0).numpy()
            flow_img = flow_viz.flow_to_image(flow_pre_cpu)
            plt.imshow(flow_img)
            plt.show()

            mask = torch.rand_like(flow_pre[0, 0]) > 0.95
            idx = torch.nonzero(mask)

            # extract correspondence points and confidence values
            mkpts0 = idx
            mkpts1 = flow_pre[0, :2, idx[:, 0], idx[:, 1]].T#mkpts0 +
            mconf = torch.full((len(mkpts0),), 0.98)



            return mkpts0, mkpts1, mconf


from TopicFM.get_matcher import get_matcher as get_matcher_TopicFM
from MatchFormer.get_matcher import get_matcher as get_matcher_MatchFormer
from LoFTR_master.get_matcher import get_matcher as get_matcher_LoFTR
from ASpanFormer.get_matcher import get_matcher as get_matcher_ASpanFormer
from FlowFormer.get_matcher import get_matcher as get_matcher_FlowFormer
tester_py = Tester_Pytorch_TopicFM(get_matcher_TopicFM(),"cuda")
tester_py.make_test(size_search=(800, 800),size_patch=(800,800),angle_rotation=0)