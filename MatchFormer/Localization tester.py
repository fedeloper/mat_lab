import pytorch_lightning as pl
import argparse
import pprint
from loguru import logger as loguru_logger
from scipy.ndimage import uniform_filter, variance
from skimage.transform import AffineTransform

from config.defaultmf import get_cfg_defaults
from model.data import MultiSceneDataModule
from model.lightning_loftr import PL_LoFTR

import PIL
from PIL import Image, ImageFilter
import findpeaks
from torch import optim
import rasterio as rio
from rasterio import warp
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
import cv2
import pytorch_lightning as pl
from matplotlib import pyplot as plt
import gc

import math
import argparse
import pprint
from distutils.util import strtobool
from pathlib import Path
from loguru import logger as loguru_logger

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


import pandas as pd

# Need this to plot in HD
# This takes up a lot of memory!
matplotlib.rcParams['figure.dpi'] = 300


def get_correspondence_multi(src_data, dst_data, src_row, src_col):
  ''' Gets the pixel coordinates (dst_row,dst_col) in dst_data of a given pixel at (src_row,src_col) in src_data.
      Warning: no handling of pixel being outside dst_data! You have to handle that yourself.
      TODO: Should add a check to only get the correspondences with high enough cornerness on both sides
      (because we should only insert meaningful correspondences in the list!).
  '''
  # Get geo coords of pixel from src dataset
  X1, Y1 = rio.transform.xy(src_data.transform, src_row, src_col)
  # print("Geographic coordinates in crs 1: ",X1,Y1)

  if src_data.crs != dst_data.crs:
    # convert coordinates in the crs of the dst dataset
    X2, Y2 = warp.transform(src_data.crs, dst_data.crs, X1, Y1)
  else:
    # if the crs is the same, do nothing
    X2, Y2 = X1, Y1

  # Get corresponding px coords in dst dataset
  # It still returns an index even if out of bounds
  dst_row, dst_col = rio.transform.rowcol(dst_data.transform, X2, Y2)
  # print("Corresponding pixel coordinates in image 2: ",dst_row,dst_col)
  return dst_row, dst_col

def get_correspondence(src_data, dst_data ,src_row, src_col):
  ''' Gets the pixel coordinates (dst_row,dst_col) in dst_data of a given pixel at (src_row,src_col) in src_data.
      Warning: no handling of pixel being outside dst_data! You have to handle that yourself.
      TODO: Should add a check to only get the correspondences with high enough cornerness on both sides
      (because we should only insert meaningful correspondences in the list!).
  '''
  # Get geo coords of pixel from src dataset
  X1, Y1 = src_data.xy(src_row, src_col)
  X1,Y1 = X1 if type(X1) is list else [X1], Y1 if type(Y1) is list else [Y1]
  #print("Geographic coordinates in crs 1: ",X1,Y1)

  if src_data.crs != dst_data.crs:
    # convert coordinates in the crs of the dst dataset
    X2, Y2 = warp.transform(src_data.crs, dst_data.crs,X1,Y1)
    X2 = X2[0]
    Y2 = Y2[0]
  else:
    # if the crs is the same, do nothing
    X2, Y2 = X1, Y1

  # Get corresponding px coords in dst dataset
  # It still returns an index even if out of bounds
  dst_row, dst_col = dst_data.index(X2, Y2)
  dst_row, dst_col =  dst_row[0] if type(dst_row) is list else dst_row, dst_col[0] if type(dst_col) is list else dst_col
  if dst_row < 0 or dst_col <0:
      print("make the margin higher, the corresponding points are outside the image")
  #print("Corresponding pixel coordinates in image 2: ",dst_row,dst_col)
  return dst_row , dst_col

def get_datapoint(ref_data,query_data):
  ''' TODO: Call this method on a pair of rasterio datasets. It will generate a random datapoint consisting
      of a smaller SAR patch from the query dataset, and a bigger SAR patch from the reference dataset.
      The search patch contains the area of the smaller patch, with a random offset.
  '''
  datapoint = None
  return datapoint

def is_good_patch(patch):
  ''' TODO: Checks if the random query patch is sufficiently texture-rich to be used to train/test the matching model.
      If the patch is too plain (e.g. sea or desert), returns False. This function should also be used at test time:
      if a sensor image is too plain, there's no need to match it.
      At test time, something similar should also be done on the reference side.
  '''
  good = None
  return good

def get_keypoints(patch):
  ''' TODO: Use something like harris corner detection to get the list of ground truth correspondences.
      In fact, you should not train on each pixel corresp, but you should select only meaningful corresp!
      Harris peaks might be just strong speckle noise, but if you take strong ones you should be fine.
  '''
  keypoints = None
  return keypoints

def normalize_image(img):
  ''' Normalize by clipping to 99th percentile and convert to uint8.
      This clips strong speckle outliers and optimizes the brightness range
  '''
  p1 = np.nanpercentile(img,99)
  img = img.clip(0,p1)
  img = (img-np.nanmin(img))/(np.nanmax(img)-np.nanmin(img))*255
  img = img.astype(np.uint8, copy=True)
  return img



import random
def get_sample(tiff1_path,tiff2_path,search_window,patch_size,margin=40,random_seed=1,verbose=True,random_rotation=0.03,random_zoom=0.03):
  '''
        1-Reads the first band of the TIFF files using the rio library and normalizes the image data.
        2-Selects a random search window and patch location within the image using random number generators.
        3-Calls the get_correspondence function on the selected locations to find the corresponding locations in the second image.
        4-Converts the grayscale images to RGB format using OpenCV.
        5-Creates centered patches from the RGB images by zero-padding the images and copying a portion of the original images to the patches.
        6-Draws rectangles around the search window and patch in both images.
        7-Saves the patch and search window as JPEG files.
        8-Returns the processed RGB images, the points of the search window and patch, and the original rio datasets.

  '''
  dataset1 = rio.open(tiff1_path)
  dataset2 = rio.open(tiff2_path)
  img1 = dataset1.read(1)
  img1 = normalize_image(img1)
  img2 = dataset2.read(1)
  img2 = normalize_image(img2)

  img1= np.swapaxes(img1,0,1)
  img2= np.swapaxes(img2,0,1)

  search_window_w,search_window_h = search_window
  patch_size_w,patch_size_h = patch_size

  if img1.shape[0]-search_window_w-margin*2 <0  or img1.shape[1]-search_window_h-margin*2 <0:
    print("margin + search windows is too big for the image:",margin,"*2 +",(search_window_w,search_window_h),">",img1.shape)
    return None,None,None,None,None,None,None
  lu = ( margin+random.randint(0,img1.shape[0]-search_window_w-margin*2),margin+random.randint(0,img1.shape[1]-search_window_h-margin*2))
  lu_patch=( lu[0] + random.randint(0,search_window_w-patch_size_w),lu[1] + random.randint(0,search_window_h-patch_size_h))


  points_patch = lu_patch
  points = lu
  #print(points,"points",img1.shape)
  points_ref = get_correspondence(dataset1, dataset2,lu[0],lu[1]  )
  points_patch_ref = get_correspondence(dataset1, dataset2,lu_patch[0] ,lu_patch[1] )

  rgb_img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
  rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

  patch_source = np.zeros((search_window_w, search_window_h), dtype = np.uint8)

  patch_source[int(search_window_w/2 - patch_size_w/2):int(search_window_w/2 + patch_size_w/2),int(search_window_h/2 - patch_size_h/2):int(search_window_h/2 + patch_size_h/2)] = img1[points_patch[0]:points_patch[0]+patch_size_w,points_patch[1]:points_patch[1]+patch_size_h]


  patch_dest = np.zeros((search_window_w, search_window_h), dtype = np.uint8)
  patch_dest[int(search_window_w/2 - patch_size_w/2):int(search_window_w/2 + patch_size_w/2),int(search_window_h/2 - patch_size_h/2):int(search_window_h/2 + patch_size_h/2)]=img2[points_patch_ref[0]:points_patch_ref[0]+patch_size_w,points_patch_ref[1]:points_patch_ref[1]+patch_size_h]



  search_window_source = img1[points[0]:points[0]+search_window_w,points[1]:points[1]+search_window_h]
  search_window_dest = img2[ points_ref[0]:points_ref[0]+search_window_w,points_ref[1] : points_ref[1]+search_window_h]


  Image.fromarray(cv2.cvtColor(patch_dest, cv2.COLOR_GRAY2RGB), "RGB").save("patch.jpeg")
  Image.fromarray(cv2.cvtColor(search_window_source, cv2.COLOR_GRAY2RGB), "RGB").save("search_window.jpeg")

    #draw searching windows




  rgb_img1  = cv2.rectangle(rgb_img1 , tuple(reversed(points)),(points[1]+search_window_h ,points[0]+search_window_w), (0,0,255), 2)
  rgb_img1  = cv2.rectangle(rgb_img1 , tuple(reversed(points_patch)), (points_patch[1]+patch_size_h, points_patch[0]+patch_size_w), (255,0,0), 2)


  #draw patch windows
  rgb_img2  = cv2.rectangle(rgb_img2 , tuple(reversed(points_ref)), ( points_ref[1]+search_window_h,points_ref[0]+search_window_w), (0,0,255), 2)
  rgb_img2  = cv2.rectangle(rgb_img2 , tuple(reversed(points_patch_ref)), ( points_patch_ref[1]+patch_size_h,points_patch_ref[0]+patch_size_w), (255,0,0), 2)

  #print(rgb_img1.shape,search_window_source.shape)


  rgb_img1= np.swapaxes(rgb_img1,0,1)
  rgb_img2= np.swapaxes(rgb_img2,0,1)
  search_window_source= np.swapaxes(search_window_source,0,1)
  patch_source= np.swapaxes(patch_source,0,1)
  search_window_dest= np.swapaxes(search_window_dest,0,1)
  patch_dest= np.swapaxes(patch_dest,0,1)
  if verbose:
      fig, axes = plt.subplots(2, 3)
      axes[0, 0].set_title('source image')
      print(rgb_img1.shape)
      axes[0, 0].imshow(PIL.ImageOps.invert(  Image.fromarray(rgb_img1)))


      axes[0, 1].set_title('search window source')
      axes[0, 1].imshow(PIL.ImageOps.invert(  Image.fromarray( cv2.cvtColor(search_window_source, cv2.COLOR_GRAY2RGB))))

      axes[0, 2].set_title('patch source')
      axes[0, 2].imshow(PIL.ImageOps.invert(  Image.fromarray(cv2.cvtColor(patch_source, cv2.COLOR_GRAY2RGB))))

      axes[1, 0].set_title('dest image')
      axes[1, 0].imshow(PIL.ImageOps.invert(  Image.fromarray(rgb_img2)))

      axes[1, 1].set_title('search window dest')
      axes[1, 1].imshow(PIL.ImageOps.invert(  Image.fromarray(cv2.cvtColor(search_window_dest, cv2.COLOR_GRAY2RGB))))

      axes[1, 2].set_title('patch dest')
      axes[1, 2].imshow(PIL.ImageOps.invert(  Image.fromarray(cv2.cvtColor(patch_dest, cv2.COLOR_GRAY2RGB))))

      plt.show()
  rgb_img1= np.swapaxes(rgb_img1,0,1)
  rgb_img2= np.swapaxes(rgb_img2,0,1)

  return rgb_img1, rgb_img2, points,points_patch_ref,points_patch,dataset1,dataset2


def get_metrics( mkpts0_r, mkpts1_r,points,points_patch,dataset1,dataset2,searching_window_w,searching_window_h,patch_w,patch_h,img0_raw,img1_raw,color,verbose=False):



                # Extract x and y coordinates from mkpts0
                x0, y0 = zip(*mkpts0_r)

                # Apply translation to the coordinates
                x_conv, y_conv = get_correspondence_multi(dataset1, dataset2, [x+points[1] for x in x0], [y+points[0] for y in y0])

                # Convert the translated coordinates back to tuples
                mk0 = list(zip(x_conv, y_conv))

                # Calculate RMSE
                rmse = 0
                for (x1, y1), (x2, y2) in zip(mk0, mkpts1_r):
                    y2 -= (searching_window_w/2)-(patch_w/2)
                    x2 -= (searching_window_h/2)-(patch_h/2)
                    y2 += points_patch[0]
                    x2 += points_patch[1]

                    rmse += ((x1-x2)**2 + (y1-y2)**2)

                # Normalize the errors by the number of keypoints
                num_kpts = len(mk0)
                rmse=  (rmse/num_kpts)**(1/2) # like in "A Transformer-Based Coarse-to-Fine Wide-Swath SAR Image Registration Method under Weak Texture Conditions"

                return rmse
import torch
from plotting import make_matching_figure
import os
def predict_and_print(rgb_img1,rgb_img2,matcher,img0_raw,img1_raw,points,points_patch,dataset1,dataset2,searching_window_w,searching_window_h,patch_w,patch_h,verbose=True,experiment=False,configs_ransac=None ):
    '''

        This code performs an image matching task with the given matcher, which takes two raw images and outputs corresponding features. The code first resizes the two raw images to (640, 480) and converts them to torch tensors, normalizing them by dividing each pixel by 255. The two images are then passed to the matcher to get feature matches and confidence scores.
        The code then computes two weighted average points based on the matches, one weighted by the confidence score and the other by a uniform weight. If the number of matches is greater than 0, the code returns the weighted average points and prints them as figures if verbose is set to True. The output figures are saved as "LoFTR-colab-demo.pdf".
I       f there are no matches, the code returns None, None, None, None.
'''


    img0 = torch.from_numpy(img0_raw)[None][None].cuda().to(torch.float) / 255.
    img1 = torch.from_numpy(img1_raw)[None][None].cuda().to(torch.float) / 255.

    batch = {'image0': img0, 'image1': img1}

    # Inference with LoFTR and get prediction
    with torch.no_grad():

            matcher.matcher(batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1= batch['mkpts1_f'].cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()

            #mkpts0 = np.array([])

    results = []
    print()
    if mkpts0.shape[0]> 4:

        if configs_ransac is not None:

            for conf in configs_ransac:
                from skimage.measure import ransac
                if conf['residual_threshold']>= 0:
                    model, inliers = ransac((mkpts0, mkpts1),AffineTransform, min_samples=conf['min_samples'], residual_threshold=conf['residual_threshold'], max_trials=conf['max_trials'])
                    n_inliers = np.sum(inliers)
                    if n_inliers is None or n_inliers< 3:
                        conf.update({'rmse':-1,'inliers':0 if n_inliers is None else n_inliers})
                       # results.append({'rmse':-1,'inliers':})
                    else:
                        rmse = get_metrics(mkpts0[inliers], mkpts1[inliers],points,points_patch,dataset1,dataset2,searching_window_w,searching_window_h,patch_w,patch_h,rgb_img1,rgb_img2,mconf[inliers],verbose=verbose)
                else:
                     rmse= get_metrics(mkpts0, mkpts1,points,points_patch,dataset1,dataset2,searching_window_w,searching_window_h,patch_w,patch_h,rgb_img1,rgb_img2,mconf,verbose=verbose)
                     n_inliers = mkpts0.shape[0]
                conf.update({'rmse':rmse,'inliers':n_inliers})
                results.append(conf)
                if conf['residual_threshold'] == 1:

                    color = cm.jet(mconf, alpha=0.7)
                    text = [
                        'LoFTR',
                        'Matches: {}'.format(len(mkpts0)),
                    ]
                    if verbose:
                        #rgb_img1= np.swapaxes(rgb_img1,0,1)
                        #rgb_img2= np.swapaxes(rgb_img2,0,1)
                        abs_m0 = np.array([(x+points[1],y+points[0]) for x,y in mkpts0[inliers]])
                        abs_m1 = np.array([(x-((searching_window_h/2)-(patch_h/2))+points_patch[1] ,y-((searching_window_w/2)-(patch_w/2))+points_patch[0]) for x,y in mkpts1[inliers]])
                        fig = make_matching_figure(rgb_img1, rgb_img2, abs_m0, abs_m1, color[inliers], abs_m0, abs_m1, text)
                        make_matching_figure(rgb_img1, rgb_img2, abs_m0, abs_m1, color[inliers], abs_m0, abs_m1, text, path="LoFTR-colab-demo.pdf")

                        fig = make_matching_figure(img0_raw, img1_raw, mkpts0[inliers], mkpts1[inliers], color[inliers], mkpts0[inliers], mkpts1[inliers], text)
                        make_matching_figure(img0_raw, img1_raw, mkpts0[inliers], mkpts1[inliers], color[inliers], mkpts0[inliers], mkpts1[inliers], text, path="LoFTR-colab-demo.pdf")
        return results
    return None

def do_test(matcher_in,path0,path1,size_search=(640, 480),size_patch=(int(180*1.333333), int(180)),configs_ransac=None,verbose=True):


  rgb_img1, rgb_img2, points,points_patch_ref,points_patch,dataset1,dataset2 = get_sample(path0,path1,size_search,size_patch,verbose=verbose)
  if rgb_img1 is None:
      return None
  img0_pth = "./search_window.jpeg"
  img1_pth = "./patch.jpeg"
  get_smoothed("search_window.jpeg","search_window_s.jpeg")
  get_smoothed("patch.jpeg","patch_s.jpeg")
  image_pair = ["search_window_s.jpeg", "patch_s.jpeg"]
  img0_raw = cv2.imread(image_pair[0], cv2.IMREAD_GRAYSCALE)
  img1_raw = cv2.imread(image_pair[1], cv2.IMREAD_GRAYSCALE)

  img0_denoised =img0_raw
  img1_denoised = img1_raw

  results = predict_and_print(rgb_img1,rgb_img2,matcher_in,img0_denoised,img1_denoised,points,points_patch,dataset1,dataset2, size_search[0],size_search[1],size_patch[0],size_patch[1],verbose=verbose,configs_ransac=configs_ransac)

  return results

def get_list():
    path_of_the_directory= '../../Data/paired_sentinel_old/'
    paths = []
    for filename in os.listdir(path_of_the_directory):
        f = os.path.join(path_of_the_directory,filename)
        if not os.path.isfile(f):
            lst = os.listdir(f)
            if len(lst)>1:
                paths.append((os.path.join(path_of_the_directory,filename,lst[0]),os.path.join(path_of_the_directory,filename,lst[1])))
    return paths