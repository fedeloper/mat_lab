import gc
import os
import pathlib
from os.path import exists
from typing import Tuple

import PIL
import cv2
import matplotlib
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.onnx.utils
from PIL import Image
from imageio import imread
from line_profiler_pycharm import profile
from matplotlib import pyplot as plt, cm

from pyproj import Transformer
from rasterio import transform as tform, MemoryFile
from rasterio import warp
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
from rasterio.transform import from_bounds, from_gcps

from scipy.ndimage import uniform_filter, variance, rotate
from torch import Tensor
from torch.optim import AdamW

from TopicFM.plotting import make_matching_figure, drawMatches

# Need thsxdfsdfis to plot in HD
# This takes up a lot of memory!

SHOW_MAGSAC_SPEED = False
SHOW_INFERENCE_SPEED = False
SHOW_CUDA_MEMORY = False

SAME_PATCH = True

matplotlib.rcParams['figure.dpi'] = 100


import numpy as np


def get_datapoint(ref_data, query_data):
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
    p1 = np.nanpercentile(img, 99)
    img = img.clip(0, p1)
    img = (img - np.nanmin(img)) / (np.nanmax(img) - np.nanmin(img)) * 255
    img = img.astype(np.uint8, copy=True)
    return img


import random



def get_sample(ph, search_window, patch_size,rnd=True):
    uavsar, llh, S1, s1_norm_log = ph

    uavsar = uavsar[:uavsar.shape[0] // 8 * 8, :uavsar.shape[1] // 8 * 8]

    s1_norm_log  = s1_norm_log[:s1_norm_log.shape[0] // 8 * 8, :s1_norm_log.shape[1] // 8 * 8]
    center_s1_norm_log = [dim // 2 for dim in s1_norm_log.shape]
    margin_more = 20
    margin_left_right = int(max((uavsar.shape[0]-search_window[0])/2,0)) + margin_more
    margin_up_bottom = int(max((uavsar.shape[1] - search_window[1]) / 2, 0)) + margin_more
    x_off = random.randrange(margin_left_right ,uavsar.shape[0]-margin_left_right-patch_size[0])
    y_off = random.randrange(margin_up_bottom  , uavsar.shape[1]-margin_up_bottom-patch_size[1])
    if not rnd:
        x_off = int((uavsar.shape[0] - patch_size[0]) / 2)
        y_off = int((uavsar.shape[1] - patch_size[1]) / 2)

    patch = uavsar[x_off: x_off + patch_size[0] ,
            y_off: y_off + patch_size[1]]

    search =s1_norm_log [center_s1_norm_log[0] - search_window[0] // 2: center_s1_norm_log[0] + search_window[0] // 2,
                     center_s1_norm_log[1] - search_window[1] // 2: center_s1_norm_log[1] + search_window[1] // 2]
    return patch, search, uavsar.shape, s1_norm_log.shape,x_off,y_off

import math

import math




def rotate_point(cx, cy, angle, x, y):
    """Rotate a point (x, y) around a center point (cx, cy) by a given angle in radians.

    Args:
        cx (float): The x-coordinate of the center point.
        cy (float): The y-coordinate of the center point.
        angle (float): The angle in radians to rotate the point around the center.
        x (list): A list of x-coordinates to rotate.
        y (list): A list of y-coordinates to rotate.

    Returns:
        A tuple of rotated x and y coordinates.
    """
    s = math.sin(angle)
    c = math.cos(angle)

    # Translate point back to origin
    x = x - cx
    y = y - cy

    # Rotate point
    xnew = x * c - y * s
    ynew = x * s + y * c

    # Translate point back to center
    xnew += cx
    ynew += cy

    return xnew, ynew




def get_correspondence(row,col,llh,S1):
    '''
    Given row and column in SLC image, get corresponding row and column in rasterio dataset.
    '''


    r,c =  S1.index(*Transformer.from_crs(4326,S1.crs.to_epsg()).transform(llh[:,:,0][row,col],llh[:, :, 1][row,col]))
    return r,c
def absolute_reference_model2dataset_uavsar(mkpts1_r,off_x,off_y):
    return mkpts1_r[:, 1] + off_x,mkpts1_r[:, 0] + off_y

def absolute_reference_model2dataset_sentinel(mkpts0_r,search_size,sh1,angle_rotation):
    # Convert the angle of rotation from degrees to radians
    angle_rad = -angle_rotation * (np.pi / 180)
    # Rotate the mkpts0_r points around the center of the search box
    i_slc, j_slc = rotate_point(search_size[0] / 2, search_size[1] / 2, angle_rad, mkpts0_r[:, 1], mkpts0_r[:, 0])
    i_slc += (sh1[0] / 2 - search_size[0] / 2)
    j_slc += (sh1[1] / 2 - search_size[1] / 2)
    return i_slc,j_slc

import _pickle as pickle


def create_geotiff_(llh_matrix, sh0, n_sample, output_filename):
    # Ensure llh_matrix and image have compatible shapes
    if llh_matrix.shape[:2] != sh0:
        raise ValueError("llh_matrix and image must have compatible shapes.")

    # Extract dimensions
    w, h = sh0

    # Create GCPs
    indices = [(i, j) for i in range(w) for j in range(h)]
    random_indices = random.sample(indices, n_sample)
    gcps = [GroundControlPoint(row=i, col=j, x=llh_matrix[i, j, 0], y=llh_matrix[i, j, 1])
            for i, j in random_indices]

    # Create transform from GCPs
    transform = from_gcps(gcps)

    # Write GeoTIFF
    with rasterio.open(output_filename, 'w', driver='GTiff', height=h, width=w, count=1, dtype=np.uint8,
                       crs=4326, transform=transform) as dst:
        return dst

@profile
def create_geotiff(llh,idx,mode='border',num_points=100000):
    # Generate arrays of random x and y indices

    if mode == 'border':
        min_lat, max_lat = np.min(llh[:, :, 0]), np.max(llh[:, :, 0])
        min_lon, max_lon = np.min(llh[:, :, 1]), np.max(llh[:, :, 1])
        # Define the pixel size in geographical units
        xres = (max_lon - min_lon) / llh.shape[1]
        yres = (max_lat - min_lat) / llh.shape[0]
        # Create transform
        transform = rasterio.transform.from_origin(min_lon, max_lat, xres, yres)
        # Write to a new raster file
        with rasterio.open(
                'output.tiff', 'w',
                driver='GTiff',
                height=llh.shape[0],
                width=llh.shape[1],
                count=1,
                dtype=np.uint8,
                crs=4326,
                transform=transform,
        ) as dst:
            return dst
    else:

            if os.path.isfile(str(idx)+".pkl"):
                with open(str(idx)+".pkl", "rb") as fp:  # Unpickling
                    transform = pickle.load(fp)
            else:
                #print("not existing",str(idx)+".pkl")
                #assert False
            # Generate arrays of random x and y indices
                x_indices = np.random.randint(0, llh.shape[0], num_points)
                y_indices = np.random.randint(0, llh.shape[1], num_points)

                # Get the corresponding lat, lon and h values
                lat_lon_h_values = llh[x_indices, y_indices]

                # Create the GroundControlPoint objects
                gcps = [GroundControlPoint(row=x, col=y, x=lon, y=lat, z=h)
                        for ((x, y), (lat, lon, h)) in zip(zip(x_indices, y_indices), lat_lon_h_values)]

                transform = from_gcps(gcps)

                with open(str(idx)+".pkl", "wb") as fp:  # Pickling
                    pickle.dump(transform, fp)
            # Write to a new raster file
            with MemoryFile().open(
                   #str(idx) + ".tiff", 'w',
                    driver='GTiff',
                    height=llh.shape[0],
                    width=llh.shape[1],
                    count=1,
                    dtype=np.uint8,
                    crs=4326,
                    transform=transform,
            ) as dst:
                return dst






#lon,lat = create_geotiff(llh, sh0).xy([0],[0])
#diffx = llh[0,0,0]-lat # comparing latitude with latitude
#diffy = llh[0,0,1]-lon # comparing longitude with longitude
def get_metrics(mkpts0_r, mkpts1_r, S1, llh, search_size, sh0, sh1, off_x, off_y,idx, angle_rotation=0,approximate=True):
    """
    Computes metrics for comparing reference points in different coordinate systems.

    Args:
        mkpts0_r: Reference model keypoints in coordinate system 0.
        mkpts1_r: Reference model keypoints in coordinate system 1.
        S1: Sentinel-1 dataset raster io.
        llh: Latitude, longitude, and height values.
        search_size: Search size for matching keypoints.
        sh0: Shape of coordinate system 0 uavsar.
        sh1: Shape of coordinate system 1 Sentinel-1.
        off_x: Offset in the x-direction for uavsar patch.
        off_y: Offset in the y-direction for uavsar patch.
        angle_rotation: Angle of rotation (default: 0).

    Returns:
        The mean distance between matched keypoints in the two coordinate systems.
    """

    # Convert reference from patch to dataset uavsar grid
    i_grd, j_grd = absolute_reference_model2dataset_uavsar(mkpts1_r, off_x, off_y)
    i_grd, j_grd = i_grd.astype(int), j_grd.astype(int)
    # Convert reference from search windows to dataset Sentinel-1 grid
    i_slc, j_slc = absolute_reference_model2dataset_sentinel(mkpts0_r, search_size, sh1, angle_rotation)

    lat, lon,h = llh[i_grd, j_grd].T


    if approximate:
        lon,lat = xy_np( create_geotiff(llh, idx,mode="CP").transform,j_grd, i_grd,)
        #lon, lat = create_geotiff(llh, idx,mode="CP").xy()
        #print("predicted", lat[0], lon[0])
        #lon1, lat1 = create_geotiff(llh, sh0, 10000, "1").xy(j_grd, i_grd)
        #print("predicted", lat1[0], lon1[0])

    i_ref, j_ref = S1.index(
        *Transformer.from_crs(4326, S1.crs.to_epsg()).transform(lat,lon))

    # Compute the mean distance between matched keypoints
    return np.hypot(i_slc - i_ref, j_slc - j_ref).mean()


def to_numpy2(transform):
    return np.array([transform.a,
    transform.b,
    transform.c,
    transform.d,
    transform.e,
    transform.f, 0, 0, 1], dtype='float64').reshape((3,3))

def to_numpy2(transform):
    return np.array([transform.a,
    transform.b,
    transform.c,
    transform.d,
    transform.e,
    transform.f, 0, 0, 1], dtype='float64').reshape((3,3))

def xy_np(transform, rows, cols, offset='center'):
    if isinstance(rows, int) and isinstance(cols, int):
        pts = np.array([[rows, cols, 1]]).T
    else:
        assert len(rows) == len(cols)
        pts = np.ones((3, len(rows)), dtype=int)
        pts[0] = rows
        pts[1] = cols

    if offset == 'center':
        coff, roff = (0.5, 0.5)
    elif offset == 'ul':
        coff, roff = (0, 0)
    elif offset == 'ur':
        coff, roff = (1, 0)
    elif offset == 'll':
        coff, roff = (0, 1)
    elif offset == 'lr':
        coff, roff = (1, 1)
    else:
        raise ValueError("Invalid offset")

    _transnp = to_numpy2(transform)
    _translt = to_numpy2(transform.translation(coff, roff))
    locs = _transnp @ _translt @ pts
    return locs[0].tolist(), locs[1].tolist()
@profile
def get_metrics_differentiable(mkpts0_r, mkpts1_r, S1, llh, search_size, sh0, sh1, off_x, off_y,idx, angle_rotation=0,approximate=False,device="cuda"):
    """
    Computes metrics for comparing reference points in different coordinate systems.

    Args:
        mkpts0_r: Reference model keypoints in coordinate system 0.
        mkpts1_r: Reference model keypoints in coordinate system 1.
        S1: Sentinel-1 dataset raster io.
        llh: Latitude, longitude, and height values.
        search_size: Search size for matching keypoints.
        sh0: Shape of coordinate system 0 uavsar.
        sh1: Shape of coordinate system 1 Sentinel-1.
        off_x: Offset in the x-direction for uavsar patch.
        off_y: Offset in the y-direction for uavsar patch.
        angle_rotation: Angle of rotation (default: 0).

    Returns:
        The mean distance between matched keypoints in the two coordinate systems.
    """
    #if mkpts1_r.isnan().any() or mkpts0_r.isnan().any():
    #    print("nan is in before metric")
    # Convert reference from patch to dataset uavsar grid
    #print(mkpts1_r.requires_grad,"m1")
    xxx =mkpts1_r[:, 1] + off_x
    yyy =mkpts1_r[:, 0] + off_y


    #i_grd, j_grd = absolute_reference_model2dataset_uavsar(mkpts1_r, off_x, off_y)

    # Convert reference from search windows to dataset Sentinel-1 grid
    i_slc, j_slc = absolute_reference_model2dataset_sentinel(mkpts0_r, search_size, sh1, angle_rotation)



    #lat, lon = S1.xy(i_slc, j_slc)
    lat, lon = xy_np(S1.transform,j_slc.detach().cpu(), i_slc.detach().cpu())
    lat,lon = Transformer.from_crs( S1.crs.to_epsg(),4326).transform(lat,lon)



    j_slc, i_slc = create_geotiff(llh, idx, mode="CP").index(lon,lat)

    grd = torch.dstack([torch.tensor(i_slc).to(device),torch.tensor(j_slc).to(device)])


    if type(xxx) is not Tensor:
        xxx = torch.tensor(xxx).to(device)
        yyy = torch.tensor(yyy).to(device)

        pred = torch.dstack([yyy, xxx])
        return torch.sqrt(torch.nn.MSELoss()(grd.float(), pred)).item()
    else:
        pred = torch.dstack([yyy, xxx])
        #print(xxx.requires_grad,pred.requires_grad,"aooooooooooooooo")
        return torch.nn.MSELoss()(grd.float(),pred)


def cornerness_friendly(gray, points,  random_select=False):

    if random_select:
        harris_near_all = np.full(points.shape[0], 500)
        # Sort and return
        result = harris_near_all.argsort()
        np.random.shuffle(result)
    else:
        corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 15)
        corners = corners.reshape(-1, 2).astype(int)

        # Select inliers only


        # Broadcast subtraction over each corner-point pair
        diffs = np.abs(points[:, None] - corners[None, :]).sum(axis=-1)

        # Get minimum difference for each point
        harris_near = diffs.min(axis=-1)

        # Create array for all points
        harris_near_all = np.full(points.shape[0], 500)

        # Sort and return
        result = harris_near_all.argsort()
        del corners,harris_near,harris_near_all,diffs

    return result


import numpy as np
import cv2

def apply_magsac_pp(mkpts0, mkpts1, mconf, configuration, searching_window_w, searching_window_h):
    if not isinstance(mkpts0, np.ndarray) or not isinstance(mkpts1, np.ndarray) or not isinstance(mconf, np.ndarray):
        raise ValueError("mkpts0, mkpts1, and mconf must be numpy arrays.")
    if len(mkpts0) != len(mkpts1) or len(mkpts0) != len(mconf):
        raise ValueError("mkpts0, mkpts1, and mconf must be of the same length.")

    order = mconf.argsort()
    mkpts0, mkpts1, mconf = mkpts0[order], mkpts1[order], mconf[order]


    if 'residual_threshold' not in configuration:
        raise ValueError("'residual_threshold' must be present in configuration dict.")


    H, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.USAC_MAGSAC, float(configuration['residual_threshold']))

    return inliers, mkpts0, mkpts1, mconf

import time
import cv2
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle

#def normalize_image(image):
 #   """
  #  Normalize the image to a 0-255 scale
  #  """
  ##  normalized_img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
   # return normalized_img

def denoise_image(image, method="bilateral"):
    """
    Denoise the image using the specified method
    """
    if method == "bilateral":
        # Bilateral filtering is more effective at removing noise while preserving edges
        denoised_img = denoise_bilateral(image, multichannel=False)
    elif method == "median":
        # Median filtering is simple and can remove salt and pepper noise
        denoised_img = median_filter(image, size=3)
    elif method == "gaussian":
        # Gaussian filtering is simple and effective for general purpose noise reduction
        denoised_img = gaussian_filter(image, sigma=1)
    elif method == "tv":
        # Total variation denoising is effective at preserving edges
        denoised_img = denoise_tv_chambolle(image, weight=0.1, multichannel=False)
    else:
        raise ValueError(f"Unsupported denoise method: {method}")

    return denoised_img

def preprocess_image(img):
    """
    Preprocess the image before keypoint detection
    """
    normalized_img = normalize_image(img)
    #denoised_img = denoise_image(normalized_img, method="bilateral") # you can change the method here
    return normalized_img

def check_valid(mkpts0,mkpts1,shape1,shape2,min_distance = 3):
    valid_indices=[]
    w0,h0 = shape1
    w1, h1 = shape2
    for i in range(mkpts0.shape[0]):
        if (0 <= mkpts0[i, 0] < w0 and 0 <= mkpts0[i, 1] < h0 and
                0 <= mkpts1[i, 0] < w1 and 0 <= mkpts1[i, 1] < h1):
            # Check if points are too close
            valid = True
            for x in valid_indices:
                distance = np.sqrt((mkpts0[i, 0] - mkpts0[x, 0]) ** 2 + (mkpts0[i, 1] - mkpts0[x, 1]) ** 2)
                if distance <= min_distance:
                    valid = False
            if valid:
                valid_indices.append(i)
    print(f"Valid points: {len(valid_indices)} / {mkpts0.shape[0]}")


from sklearn.metrics import mean_squared_error

def postproces_and_metric(conf, mkpts0, mkpts1, img0, img1, ph, searching_window_w,
                          searching_window_h, patch_w, patch_h, sh0, sh1, mconf, threshold_inliers,
                          random_select, off_x,off_y,name_model,idx,
                          save_pdf=True, verbose=False, angle_rotation=0):

    if conf['residual_threshold'] >= 0 and mkpts0.shape[0]>4:
        if SHOW_MAGSAC_SPEED:
            st = time.time()

        inliers, mkpts0, mkpts1, mconf = apply_magsac_pp(mkpts0, mkpts1, mconf, conf, searching_window_w, searching_window_h)

        if SHOW_MAGSAC_SPEED:
            et = time.time()
            elapsed_time = et - st
            print('Magsac Execution time:', elapsed_time, 's seconds inliers:', np.sum(inliers))
    else:
        inliers = np.zeros_like(range(mkpts0.shape[0]))
        inliers[:] = True
    inliers=inliers.astype(bool)
    threshold_inliers = threshold_inliers if threshold_inliers >= 0 else np.sum(inliers)
    n_inliers = np.sum(inliers)

    if inliers is None or n_inliers == 0 or n_inliers < threshold_inliers:
        conf.update({'rmse': -1, 'inliers': 0 if n_inliers is None else n_inliers})
    else:
        gray = img0.squeeze()

        mkpts0 = mkpts0.squeeze()
        mkpts1 = mkpts1.squeeze()
        inliers= inliers.squeeze()
        mk0_in = mkpts0[inliers]
        mk1_in = mkpts1[inliers]

        m_conf_in = mconf[inliers]

        new_order = cornerness_friendly(gray,mk0_in , random_select)

        mk0_in = mk0_in[new_order ]
        mk1_in = mk1_in[new_order ]
        m_conf_in = m_conf_in[new_order ]
        uavsar, llh, S1, s1_norm_log = ph



        rmse = get_metrics_differentiable(np.array(mk0_in, copy=True)[:threshold_inliers],np.array(mk1_in, copy=True)[:threshold_inliers], S1,llh, (searching_window_w, searching_window_h),   sh0=sh0, sh1=sh1, off_x=off_x, off_y=off_y,idx=idx, angle_rotation=angle_rotation)

        conf.update({'rmse': rmse, 'inliers': n_inliers})
        save_pdf=False
        if save_pdf:
            color = cm.jet(m_conf_in[:threshold_inliers], alpha=0.7)

            if verbose:

               # drawMatches(img0, mk0_in[:threshold_inliers].squeeze(), img1, mk1_in[:threshold_inliers].squeeze())
                make_matching_figure(img1.squeeze(), img0.squeeze(), mk1_in[:threshold_inliers].squeeze(),
                                     mk0_in[:threshold_inliers].squeeze(), color, path=name_model.split("/")[0] + "/" + name_model.split("/")[
                                         1] + "_thr_" + str(conf['residual_threshold']) + "_rmse_" + str(
                                         rmse) + "_inliers_" + str(n_inliers) + ".pdf")

            del color
        del gray
    return conf

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_TRANSPARENT)
  return result

import cv2
import numpy as np
from scipy.ndimage.filters import convolve


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
def predict(matcher, img0, img1, angle_rotation=0, kernel_size=3):
    '''
    This code performs an image matching task with the given matcher, which takes two raw images and outputs corresponding features.
    The code first resizes the two raw images to (640, 480) and converts them to torch tensors, normalizing them by dividing each pixel by 255.
    The two images are then passed to the matcher to get feature matches and confidence scores. The code then computes two weighted average points based on the matches,
    one weighted by the confidence score and the other by a uniform weight.
    If the number of matches is greater than 0, the code returns the weighted average points and prints them as figures if verbose is set to True. The output figures are saved as "LoFTR-colab-demo.pdf".
    If there are no matches, the code returns None, None, None, None.
    '''


    #img0 = cv2.GaussianBlur(img0, (kernel_size, kernel_size), 0)
    #img1 = cv2.GaussianBlur(img1, (kernel_size, kernel_size), 0)
    img1 = rotate_image(img1, -angle_rotation)

    img0 = img0[None, None] / 255.
    img1 = img1[None, None] / 255.





    with torch.no_grad():
        torch.cuda.reset_peak_memory_stats(device="cuda")

        if SHOW_INFERENCE_SPEED:
            import time
            st = time.time()
        #try:
        mkpts0, mkpts1, mconf = matcher(img0, img1)
        #except:
        #    return None
       # print("chec valid",1)
       # check_valid(mkpts0,mkpts1,sh0,sh1)
        if SHOW_INFERENCE_SPEED:
            print('Single inference Execution time:', time.time() - st, "s")

        if SHOW_CUDA_MEMORY:
            import psutil
            print(psutil.virtual_memory())
            print("max memory allocated cuda", torch.cuda.max_memory_allocated(device="cuda"))

        return (mkpts0, mkpts1, mconf) if mkpts0.shape[0] > 0 else None



def lee_filter(img, size):
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img ** 2, (size, size))
    img_variance = img_sqr_mean - img_mean ** 2

    overall_variance = variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output


def do_test(matcher_in, ph, idx,size_search=(640, 480), size_patch=(int(180 * 1.333333), int(180)),name_model="Noname", threshold_inliers=5,
            configs_ransac=None, verbose=True, random_select=False,angle_rotation=0):

    patch,search,sh0,sh1,off_x,off_y = get_sample(ph, size_search, size_patch,rnd=False)
    results = None
    prediction = predict( matcher_in, search,patch,angle_rotation=angle_rotation)

    if prediction is not None:
        mkpts0, mkpts1, mconf= prediction


        results =  [postproces_and_metric(conf, mkpts0, mkpts1, search,patch, ph,
                                  size_search[0], size_search[1], size_patch[0], size_patch[1], sh0,sh1,
                                  mconf, threshold_inliers, random_select,off_x,off_y,idx=idx, name_model=name_model, verbose=verbose,
                                  angle_rotation=angle_rotation) for conf in
            configs_ransac]
        del mkpts0,mkpts1,mconf
    del patch,search,prediction

    return results




def do_pyramidal_test(matcher_in, ph,starting_size_search, size_search=(640, 480), size_patch=(int(180 * 1.333333), int(180)),name_model="Noname", threshold_inliers=5,
            configs_ransac=None, verbose=True, random_select=False,angle_rotation=0, pyramidal_steps=10):




    patch_dest, search_window_source, rgb_img1, rgb_img2, points, points_patch_ref, points_patch = get_sample(ph,
                                                                                                              starting_size_search,
                                                                                                              size_patch,
                                                                                                                margin= 60,
                                                                                                           verbose=verbose)
    configuration = configs_ransac[3]
    if rgb_img1 is None:
        return None


    step = (1-(size_search[0]/starting_size_search[0]))/pyramidal_steps
    new_corner = (0,0)
    for count in range(0,pyramidal_steps):
        factor = (size_search[0]/starting_size_search[0]) + step*(pyramidal_steps-count)

        print(new_corner)
        partial_search = search_window_source[int(new_corner[0]):int(new_corner[0]+ starting_size_search[0] * (factor)) ,int(new_corner[1]):int(new_corner[1]+ starting_size_search[1] * (factor) )]
        print("shape partial",partial_search.shape)
        prediction = predict(matcher_in,cv2.resize(partial_search, size_search),  patch_dest ,angle_rotation=angle_rotation)

        if prediction is not None:
            mkpts0,mkpts1,mconf,img0,img1 = prediction
            inliers, mkpts0, mkpts1, mconf = apply_magsac_pp( mkpts0,mkpts1,mconf,configuration,size_search[0], size_search[1])
            mkpts0, mkpts1, mconf = mkpts0[inliers], mkpts1[inliers], mconf[inliers]
            #print(mkpts0)
            distance_from_left_upper = (mkpts1[0][0],mkpts1[0][1])
            if count<pyramidal_steps-1:
                new_corner = (new_corner[0]+(mkpts0[0][0]*(1/factor)-distance_from_left_upper[0]),new_corner[1]+(mkpts0[0][1]*(1/factor)-distance_from_left_upper[1]))
                new_corner = (min(-new_corner[0], starting_size_search[0] - starting_size_search[0] * (factor - step)),
                              min(-new_corner[1], starting_size_search[1] - starting_size_search[1] * (factor - step)))



    for x in range(len(mkpts0)):
        mkpts0[x][0] = mkpts0[x][0] * (1 / factor) - new_corner[0]
        mkpts0[x][1] = mkpts0[x][1] * (1 / factor) - new_corner[1]

    rmse= get_metrics(mkpts0, mkpts1, points, points_patch, points_patch_ref, ph, size_search[0],size_search[1],
                size_patch[0], size_patch[1], angle_rotation)
    #print("rmse ", rmse)

def get_list():
    path_of_the_directory = '../Tester/data/dataset/'
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


############## SENTINEL-1 PREPROCESSING FUNCTIONS ##############

def normalize_sar(band, percentile):
    p = np.nanpercentile(band, percentile)
    img = band.clip(0, p)
    img_n = (img - np.nanmin(img)) / (np.nanmax(img) - np.nanmin(img))
    img_n[img_n == np.nan] = 0
    return img_n


############## SENTINEL-2 PREPROCESSING FUNCTIONS ##############

def normalize(band):
    band_min, band_max = (np.nanmin(band), np.nanmax(band))
    return ((band - band_min) / ((band_max - band_min)))


def gammacorr(band):
    gamma = 2
    return np.power(band, 1 / gamma)


def create_RGB_composite(red, green, blue):
    red_g = gammacorr(red)
    blue_g = gammacorr(blue)
    green_g = gammacorr(green)

    red_gn = normalize(red_g)
    green_gn = normalize(green_g)
    blue_gn = normalize(blue_g)

    rgb_composite = np.dstack((red_gn, green_gn, blue_gn))
    return rgb_composite

def calculate_angle_p(lat, long):
        # Convert latitude and longitude to radians
        lat_rad = np.radians(lat)
        long_rad = np.radians(long)

        # Calculate the differences in latitude and longitude
        delta_lat = lat_rad[1:, :] - lat_rad[:-1, :]
        delta_long = long_rad[1:, :] - long_rad[:-1, :]

        # Calculate the angle with respect to the north
        angle_rad = np.arctan2(delta_long, delta_lat)
        angle_deg = np.degrees(angle_rad)

        # Adjust the angle to be between 0 and 360 degrees
        angle_deg = np.mod(angle_deg + 360, 360)

        return angle_deg


def calculate_angle(lat1, long1, lat2, long2):
    # Convert latitude and longitude to radians

    angle1 = np.mean(calculate_angle_p(lat1, long1))
    angle2 = np.mean(calculate_angle_p(lat2, long2))

    return angle1 - angle2


def rotate_map(lat1, lon1, map2, lat2, lon2):
    diff_angle = calculate_angle(lat1[int(lat1.shape[0] / 2 - 300):int(lat1.shape[0] / 2 + 300),
                                 int(lat1.shape[1] / 2 - 300):int(lat1.shape[1] / 2 + 300)],
                                 lon1[int(lat1.shape[0] / 2 - 300):int(lat1.shape[0] / 2 + 300),
                                 int(lat1.shape[1] / 2 - 300):int(lat1.shape[1] / 2 + 300)],
                                 lat2[int(lat2.shape[0] / 2 - 300):int(lat2.shape[0] / 2 + 300),
                                 int(lat2.shape[1] / 2 - 300):int(lat2.shape[1] / 2 + 300)],
                                 lon2[int(lat2.shape[0] / 2 - 300):int(lat2.shape[0] / 2 + 300),
                                 int(lat2.shape[1] / 2 - 300):int(lat2.shape[1] / 2 + 300)])
    # print(diff_angle)
    # diff_angle = rotation_angle1-rotation_angle2 #- math.degrees(math.pi/32)
    # print(rotation_angle1,rotation_angle2,diff_angle)
    diff_angle += np.degrees(math.pi / 40)
    rotated_lat2 = rotate(lat2, diff_angle)
    rotated_lon2 = rotate(lon2, diff_angle)

    rotated_map2 = rotate(map2, diff_angle)

    return rotated_lat2, rotated_lon2, rotated_map2

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import CRS, Transformer
from scipy.ndimage import rotate


import numpy as np
import rasterio
from scipy.ndimage import rotate




from PIL import Image
import numpy as np
def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
  return result
from PIL import Image
import numpy as np
@profile
def rotate_llh_map(img, llh, geotif_path):
    # Read the GeoTIFF file
    with rasterio.open(geotif_path) as geotif:
        geotif_transform = geotif.transform

    # Calculate the angle of the GeoTIFF respect to north
    dy_geotif = geotif_transform.f
    dx_geotif = geotif_transform.e
    angle_geotif = np.degrees(np.arctan2(dy_geotif, dx_geotif))

    # Calculate the angle of the llh/map respect to north
    dy_llh = llh[500, -1, 1] - llh[0, 500, 1]
    dx_llh = llh[-1, 500, 0] - llh[0, 500, 0]
    angle_llh = np.degrees(np.arctan2(dy_llh, dx_llh))

    # Calculate the difference in angles
    angle_diff = angle_geotif - angle_llh + 90

    # Convert the NumPy array to PIL Image
    #img_pil = Image.fromarray(img)

    # Rotate the image
    img_rotated = rotate_image(img,angle_diff)#img_pil.rotate(angle_diff, resample=Image.BICUBIC, expand=False)

    # Convert the PIL Image back to NumPy array
    img_rotated = np.array(img_rotated)

    # Rotate each channel of llh
    llh_rotated =rotate_image(llh,angle_diff)
    #for i in range(llh.shape[2]):
     #   llh_channel = llh[:, :, i]
     #   llh_channel_rotated = Image.fromarray(llh_channel).rotate(angle_diff, resample=Image.BICUBIC, expand=False)
     #   llh_rotated[:, :, i] = np.array(llh_channel_rotated)

    return llh_rotated, img_rotated
