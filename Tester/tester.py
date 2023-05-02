import os
import pathlib
from typing import Tuple

import PIL
import cv2
import matplotlib
import numpy as np
import rasterio
import torch
import torch.onnx.utils
from PIL import Image
from matplotlib import pyplot as plt, cm
from rasterio import transform as tform
from rasterio import warp
from rasterio.crs import CRS
# from pymagsac import pymagsac
from scipy.ndimage import uniform_filter, variance

from TopicFM.plotting import make_matching_figure



# Need thsxdfsdfis to plot in HD
# This takes up a lot of memory!

SHOW_MAGSAC_SPEED = False
SHOW_INFERENCE_SPEED = True
SHOW_CUDA_MEMORY = False
SHOW_TOTAL_IMAGEINVERBOSE = False
SAME_PATCH = True

matplotlib.rcParams['figure.dpi'] = 100


import numpy as np

def get_correspondence(i, j, llh_array, transform, crs):
    ''' Given rows and columns in SLC image, get corresponding rows and columns in rasterio dataset.
    @param i: list of rows in slc.
    @param j: list of columns in slc.
    @param llh_array: the llh as a dictionary of numpy arrays (one per lat, long, height).
    @param transform: the geotransform of the rasterio dataset.
    @param crs: the crs of the rasterio dataset.
    @return i_ref: list of rows in grd.
    @return j_ref: list of columns in grd.
    '''
    if type(i) is not list:
        i = [i]
        j = [j]

    i = np.round(i).astype(int)
    j = np.round(j).astype(int)
    lat = llh_array[f'llh.lat'][i, j]
    lon = llh_array[f'llh.long'][i, j]
    # Convert the EPSG:4326 coordinate to the CRS of the raster
    X, Y = warp.transform(crs, CRS.from_string("EPSG:4326"), lon, lat)
    coords = np.array([X, Y])
    # Calculate the corresponding pixel coordinates
    j_ref, i_ref = ~transform * coords
    i_ref = np.round(i_ref).astype(int)
    j_ref = np.round(j_ref).astype(int)
    return i_ref, j_ref

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



def get_sample(ph, search_window, patch_size, margin=90, random_seed=1, verbose=True,
               random_rotation=0.03, random_zoom=0.03, inner_margin=30):
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
    p1, crs_path, p2, hhl_path = ph
    img1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.resize(img1, (img1.shape[1] // 8 * 8, img1.shape[0] // 8 * 8))  # input size shuold be divisible by 8
    img2 = cv2.resize(img2, (img2.shape[1] // 8 * 8, img2.shape[0] // 8 * 8))

    img1 = np.swapaxes(img1, 0, 1)
    img2 = np.swapaxes(img2, 0, 1)

    search_window_w, search_window_h = search_window
    patch_size_w, patch_size_h = patch_size

    if img1.shape[0] - search_window_w - margin * 2 < 0 or img1.shape[1] - search_window_h - margin * 2 < 0:
        print("margin + search windows is too big for the image:", margin, "*2 +", (search_window_w, search_window_h),
              ">", img1.shape)
        return None, None, None, None, None, None, None
    lu = (margin + 0 if SAME_PATCH else random.randint(0, img1.shape[0] - search_window_w - margin * 2),
          margin + 0 if SAME_PATCH else random.randint(0, img1.shape[1] - search_window_h - margin * 2))
    rd = (lu[0] + search_window_w, lu[1] + search_window_h)
    lu_patch = (lu[0] + inner_margin + 0 if SAME_PATCH else random.randint(0, search_window_w - patch_size_w - 2 * inner_margin),
                lu[1] + inner_margin + 0 if SAME_PATCH else random.randint(0, search_window_h - patch_size_h - 2 * inner_margin))
    rd_patch = (lu_patch[0] + search_window_w, lu_patch[1] + search_window_h)

    points_patch = lu_patch
    points = lu
    # print(points,"points",img1.shape)
    p1, crs_path, p2, hhl_path = ph
    # load llh
    with open(hhl_path, 'rb') as f:
        query_lat = np.load(f)
        query_long = np.load(f)
    llh = {}
    llh[f'llh.lat'] = query_lat
    llh[f'llh.long'] = query_long

    # load geotransform
    with open(crs_path, 'rb') as f:
        trans = np.load(f)
        trans = tform.Affine(*trans.flatten()[:6])
        crs = CRS.from_epsg(np.load(f))

    points_ref = get_correspondence(lu[0], lu[1], llh, trans, crs)[0][0],get_correspondence(lu[0], lu[1], llh, trans, crs)[1][0]
    points_patch_ref = get_correspondence(lu_patch[0], lu_patch[1], llh, trans, crs)[0][0], get_correspondence(lu_patch[0], lu_patch[1], llh, trans, crs)[1][0]

    rgb_img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

    patch_source = np.zeros((search_window_w, search_window_h), dtype=np.uint8)

    patch_source[int(search_window_w / 2 - patch_size_w / 2):int(search_window_w / 2 + patch_size_w / 2),
    int(search_window_h / 2 - patch_size_h / 2):int(search_window_h / 2 + patch_size_h / 2)] = img1[points_patch[0]:
                                                                                                    points_patch[
                                                                                                        0] + patch_size_w,
                                                                                               points_patch[1]:
                                                                                               points_patch[
                                                                                                   1] + patch_size_h]

    patch_dest = np.zeros((search_window_w, search_window_h), dtype=np.uint8)
    patch_dest[int(search_window_w / 2 - patch_size_w / 2):int(search_window_w / 2 + patch_size_w / 2),
    int(search_window_h / 2 - patch_size_h / 2):int(search_window_h / 2 + patch_size_h / 2)] = img2[points_patch_ref[0]:
                                                                                                    points_patch_ref[
                                                                                                        0] + patch_size_w,
                                                                                               points_patch_ref[1]:
                                                                                               points_patch_ref[
                                                                                                   1] + patch_size_h]

    search_window_source = img1[points[0]:points[0] + search_window_w, points[1]:points[1] + search_window_h]
    search_window_dest = img2[points_ref[0]:points_ref[0] + search_window_w,
                         points_ref[1]: points_ref[1] + search_window_h]

    # draw searching windows

    rgb_img1 = cv2.rectangle(rgb_img1, tuple(reversed(points)),
                             (points[1] + search_window_h, points[0] + search_window_w), (0, 0, 255), 2)
    rgb_img1 = cv2.rectangle(rgb_img1, tuple(reversed(points_patch)),
                             (points_patch[1] + patch_size_h, points_patch[0] + patch_size_w), (255, 0, 0), 2)

    # draw patch windows
    rgb_img2 = cv2.rectangle(rgb_img2, tuple(reversed(points_ref)),
                             (points_ref[1] + search_window_h, points_ref[0] + search_window_w), (0, 0, 255), 2)
    rgb_img2 = cv2.rectangle(rgb_img2, tuple(reversed(points_patch_ref)),
                             (points_patch_ref[1] + patch_size_h, points_patch_ref[0] + patch_size_w), (255, 0, 0), 2)

    # print(rgb_img1.shape,search_window_source.shape)

    rgb_img1 = np.swapaxes(rgb_img1, 0, 1)
    rgb_img2 = np.swapaxes(rgb_img2, 0, 1)
    search_window_source = np.swapaxes(search_window_source, 0, 1)
    patch_source = np.swapaxes(patch_source, 0, 1)
    search_window_dest = np.swapaxes(search_window_dest, 0, 1)
    patch_dest = np.swapaxes(patch_dest, 0, 1)
    if verbose:
        fig, axes = plt.subplots(2, 3)
        axes[0, 0].set_title('source image')
        # print(rgb_img1.shape)
        axes[0, 0].imshow(PIL.ImageOps.invert(Image.fromarray(rgb_img1)))

        axes[0, 1].set_title('search window source')
        axes[0, 1].imshow(PIL.ImageOps.invert(Image.fromarray(cv2.cvtColor(search_window_source, cv2.COLOR_GRAY2RGB))))

        axes[0, 2].set_title('patch source')
        axes[0, 2].imshow(PIL.ImageOps.invert(Image.fromarray(cv2.cvtColor(patch_source, cv2.COLOR_GRAY2RGB))))

        axes[1, 0].set_title('dest image')
        axes[1, 0].imshow(PIL.ImageOps.invert(Image.fromarray(rgb_img2)))

        axes[1, 1].set_title('search window dest')
        axes[1, 1].imshow(PIL.ImageOps.invert(Image.fromarray(cv2.cvtColor(search_window_dest, cv2.COLOR_GRAY2RGB))))

        axes[1, 2].set_title('patch dest')
        axes[1, 2].imshow(PIL.ImageOps.invert(Image.fromarray(cv2.cvtColor(patch_dest, cv2.COLOR_GRAY2RGB))))

        plt.show()
    rgb_img1 = np.swapaxes(rgb_img1, 0, 1)
    rgb_img2 = np.swapaxes(rgb_img2, 0, 1)
    patch_dest = np.swapaxes(patch_dest, 0, 1)
    search_window_source = np.swapaxes(search_window_source, 0, 1)

    return patch_dest, search_window_source, rgb_img1, rgb_img2, points, points_patch_ref, points_patch

def rotate_list_of_prediction(pred):
    for i in range(len(pred)):
        x,y= pred
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
    x -= cx
    y -= cy

    # Rotate point
    xnew = x * c - y * s
    ynew = x * s + y * c

    # Translate point back to center
    xnew += cx
    ynew += cy

    return xnew, ynew


def get_metrics(mkpts0_r, mkpts1_r, points, points_patch, points_patch_ref, ph, searching_window_w, searching_window_h,
                patch_w, patch_h, angle_rotation):
    p1, crs_path, p2, hhl_path = ph
    # load llh
    with open(hhl_path, 'rb') as f:
        query_lat = np.load(f)
        query_long = np.load(f)
    llh = {}
    llh[f'llh.lat'] = query_lat
    llh[f'llh.long'] = query_long

    # load geotransform
    with open(crs_path, 'rb') as f:
        trans = np.load(f)
        trans = tform.Affine(*trans.flatten()[:6])
        crs = CRS.from_epsg(np.load(f))

    # Extract x and y coordinates from mkpts0
    x0, y0 = zip(*mkpts0_r)

    # Apply translation to the coordinates
    i_slc, j_slc = rotate_point(searching_window_w / 2, searching_window_h / 2, -angle_rotation * (math.pi / 180), mkpts1_r[:, 1], mkpts1_r[:, 0])
    i_grd, j_grd = np.round(mkpts0_r[:, 1]).astype(int), np.round(mkpts0_r[:, 0]).astype(int)  # SEARCH - SAT
    i_grd += points[0]
    j_grd += points[1]

    j_slc = j_slc - ((searching_window_h / 2) - (patch_h / 2)) + points_patch_ref[1]
    i_slc = i_slc - ((searching_window_w / 2) - (patch_w / 2)) + points_patch_ref[0]

    # Given points in SLC, get correspondence in GRD
    j_ref, i_ref = get_correspondence(j_slc.tolist(), i_slc.tolist(), llh, trans, crs)

    rmse = np.sqrt(np.mean((i_grd - i_ref)**2 + (j_grd - j_ref)**2))

    return rmse

def cornerness_friendly(gray, points, inliers, random_select=False):
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 15)

    corners = np.int0(corners)
    harris_near = np.zeros((points.shape[0])) + 500

    for match in range(len(points)):
        if inliers[match]:
            arr = [abs(points[match][0] - corner.ravel()[0]) + abs(points[match][1] - corner.ravel()[1]) for corner in
                   corners]
            harris_near[match] = min(arr)
    result = harris_near.argsort()
    if random_select:
        random.shuffle(result)
    return result

import pymagsac

def postproces_and_metric(conf, mkpts0, mkpts1, img0,img1, ph, points, points_patch, points_patch_ref, searching_window_w,
                          searching_window_h, patch_w, patch_h, rgb_img1, rgb_img2, mconf, threshold_inliers,
                          random_select,name_model,
                          save_pdf=True, verbose=False,angle_rotation=0):
    if conf['residual_threshold'] >= 0:

        if SHOW_MAGSAC_SPEED:
            import time
            # get the start time
            st = time.time()
        order = (mconf).argsort()
        mkpts0 = mkpts0[order]
        mkpts1 = mkpts1[order]
        mconf = mconf[order]

        probabilities = mconf + min(mconf)
        probabilities /= max(probabilities)

        correspondences = np.float32([np.concatenate((mkpts0[m], mkpts1[m])) for m in range(len(mkpts0))]).reshape(-1,
                                                                                                                   4)
        H, inliers = pymagsac.findHomography(
            np.ascontiguousarray(correspondences),
            searching_window_w, searching_window_h, searching_window_w, searching_window_h,
            probabilities=probabilities,
            sampler=4,
            # conf = 0.,
            use_magsac_plus_plus=False,
            min_iters=conf['max_trials'],
            max_iters=conf['max_trials'],
            sigma_th=float(conf['residual_threshold']))

        if SHOW_MAGSAC_SPEED:
            et = time.time()
            # get the execution time
            elapsed_time = et - st
            print('Magsac Execution time:', elapsed_time, 's seconds inliers:', np.sum(inliers))

    else:
        inliers = np.zeros_like(range(mkpts0.shape[0]))
        inliers[:] = True

    threshold_inliers = threshold_inliers if threshold_inliers >= 0 else np.sum(inliers)
    n_inliers = np.sum(inliers)

    if inliers is None or n_inliers is None or n_inliers == 0 or (n_inliers < threshold_inliers):
        conf.update({'rmse': -1, 'inliers': 0 if n_inliers is None else n_inliers})
    else:
        gray = img0.squeeze()
        new_order = cornerness_friendly(gray, mkpts0, inliers, random_select)

        inliers = inliers[new_order]
        # print(mkpts0.shape,mkpts0[new_order][inliers][:threshold_inliers].shape)

        rmse = get_metrics(mkpts0[new_order][inliers][:threshold_inliers],
                           mkpts1[new_order][inliers][:threshold_inliers], points,
                           points_patch, points_patch_ref, ph,
                           searching_window_w, searching_window_h, patch_w, patch_h,angle_rotation=angle_rotation)
        conf.update({'rmse': rmse, 'inliers': n_inliers})
        p1, crs_path, p2, hhl_path = ph
        # load llh


        if save_pdf:

            color = cm.jet(mconf[new_order][inliers][:threshold_inliers], alpha=0.7)
            abs_m0 = np.array(
                [(x + points[1], y + points[0]) for x, y in mkpts0[new_order][inliers][:threshold_inliers]])
            abs_m1 = np.array([(x - ((searching_window_h / 2) - (patch_h / 2)) + points_patch_ref[1],
                                y - ((searching_window_w / 2) - (patch_w / 2)) + points_patch_ref[0]) for x, y in
                               mkpts1[new_order][inliers][:threshold_inliers]])
            # abs_m0 = np.array([(get_correspondence(abs_m1[0][0],abs_m1[0][1],llh, trans, crs))])
            text = [
                str(conf),
                'Matches: {}'.format(len(mkpts0[new_order][inliers][:threshold_inliers])),
                "x - ((searching_window_h / 2) - (patch_h / 2)) + points_patch_ref[1]",
                "x - ((" + str(searching_window_h) + str("/2)-") + str(patch_h) + str(" / 2)) + ") + str(
                    points_patch_ref[1]),
                "y - ((searching_window_w / 2) - (patch_w / 2)) + points_patch_ref[0])",
                "y - ((" + str(searching_window_w) + str("/2)-") + str(patch_w) + str(" / 2)) + ") + str(
                    points_patch_ref[0]),
            ]

            #if conf['residual_threshold'] <= 0.5:
            if verbose:
                if SHOW_TOTAL_IMAGEINVERBOSE:
                    fig = make_matching_figure(rgb_img1, rgb_img2, abs_m0, abs_m1, color, abs_m0[:1], abs_m1,
                                           text)

                else:
                    fig = make_matching_figure(img0.squeeze(), img1.squeeze(), mkpts0[new_order][inliers][:threshold_inliers], mkpts1[new_order][inliers][:threshold_inliers], color, mkpts0[new_order][inliers][:threshold_inliers], mkpts1[new_order][inliers][:threshold_inliers],
                                               text)

            if SHOW_TOTAL_IMAGEINVERBOSE:
                make_matching_figure(rgb_img1, rgb_img2, abs_m0, abs_m1, color, abs_m0, abs_m1, text,
                                     path=name_model.split("/")[0] + "/" + name_model.split("/")[
                                         1] + "_thr_" + str(conf['residual_threshold']) + "_rmse_" + str(
                                         rmse) + "_inliers_" + str(n_inliers) + ".pdf")
            else:
                make_matching_figure(img0.squeeze(), img1.squeeze(),
                                     mkpts0[new_order][inliers][:threshold_inliers],
                                     mkpts1[new_order][inliers][:threshold_inliers], color,
                                     mkpts0[new_order][inliers][:threshold_inliers],
                                     mkpts1[new_order][inliers][:threshold_inliers],
                                     text,path=name_model.split("/")[0] + "/" + name_model.split("/")[
                                         1] + "_thr_" + str(conf['residual_threshold']) + "_rmse_" + str(
                                         rmse) + "_inliers_" + str(n_inliers) + ".pdf")
    return conf

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def predict_and_print(ph, rgb_img1, rgb_img2, matcher, img0_raw, img1_raw, points, points_patch, points_patch_ref,
                      searching_window_w, searching_window_h, patch_w, patch_h, configs_ransac, threshold_inliers,
                      random_select=False, verbose=False,name_model="No_Name",angle_rotation=0):
    '''
        This code performs an image matching task with the given matcher, which takes two raw images and outputs corresponding features. The code first resizes the two raw images to (640, 480) and converts them to torch tensors, normalizing them by dividing each pixel by 255. The two images are then passed to the matcher to get feature matches and confidence scores.
        The code then computes two weighted average points based on the matches, one weighted by the confidence score and the other by a uniform weight. If the number of matches is greater than 0, the code returns the weighted average points and prints them as figures if verbose is set to True. The output figures are saved as "LoFTR-colab-demo.pdf".
I       f there are no matches, the code returns None, None, None, None.
'''

    img0 = (torch.from_numpy(img0_raw)[None][None].to(torch.float) / 255.).numpy()
    img1_raw = rotate_image(img1_raw,angle_rotation)
    img1 = (torch.from_numpy(img1_raw)[None][None].to(torch.float) / 255.).numpy()

    # batch = {'image0': img0, 'image1': img1}

    # Inference with LoFTR and get prediction
    with torch.no_grad():
        torch.cuda.reset_peak_memory_stats(device="cuda")

        if SHOW_INFERENCE_SPEED:
            import time

            # get the start time
            st = time.time()

        mkpts0, mkpts1, mconf = matcher(img0, img1)
        if SHOW_INFERENCE_SPEED:
            et = time.time()
            # get the execution time
            elapsed_time = et - st
            print('Single inference Execution time:', elapsed_time, "s")
        if SHOW_CUDA_MEMORY:
            print("max memory allocated cuda", torch.cuda.max_memory_allocated(device="cuda"))
        mkpts0 = mkpts0
        mkpts1 = mkpts1
        mconf = mconf

    if mkpts0.shape[0] > 0:
        return [postproces_and_metric(conf, mkpts0, mkpts1, img0,img1, ph, points, points_patch, points_patch_ref,
                                      searching_window_w, searching_window_h, patch_w, patch_h, rgb_img1, rgb_img2,
                                      mconf, threshold_inliers, random_select,name_model=name_model, verbose=verbose,angle_rotation=angle_rotation) for conf in
                configs_ransac]

    return None


def lee_filter(img, size):
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img ** 2, (size, size))
    img_variance = img_sqr_mean - img_mean ** 2

    overall_variance = variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output


def do_test(matcher_in, ph, size_search=(640, 480), size_patch=(int(180 * 1.333333), int(180)),name_model="Noname", threshold_inliers=5,
            configs_ransac=None, verbose=True, random_select=False,angle_rotation=0):
    patch_dest, search_window_source, rgb_img1, rgb_img2, points, points_patch_ref, points_patch = get_sample(ph,
                                                                                                              size_search,
                                                                                                              size_patch,
                                                                                                              verbose=verbose)
    if rgb_img1 is None:
        return None

    import skimage.measure

    results = predict_and_print(ph, rgb_img1, rgb_img2, matcher_in, search_window_source, patch_dest, points,
                                points_patch, points_patch_ref,
                                size_search[0], size_search[1], size_patch[0], size_patch[1]
                                , configs_ransac, threshold_inliers, random_select,name_model=name_model, verbose=verbose,angle_rotation=angle_rotation
                                )

    entropy = skimage.measure.shannon_entropy(patch_dest)

    return results, entropy


def get_list():
    path_of_the_directory = './data/UAVSAR/'
    paths = []
    for filename in os.listdir(path_of_the_directory):
        f = os.path.join(path_of_the_directory, filename)
        if len(os.listdir(f)) > 3:
            if not os.path.isfile(f):
                lst = sorted([os.path.abspath(os.path.join(f, p)) for p in os.listdir(f)])
                lst = [item for item in lst if "az_rg" not in item]
                paths.append(lst)

    return paths


class Tester():
    def __init__(self):
        self.name_model= "No_Name"


    def make_inference(self, image0: np.numarray, image1: np.numarray) -> Tuple[np.numarray, np.numarray, np.numarray]:
        return (np.array([]), np.array([]), np.array([]))

    def make_test(self, size_search=(800, 720),
                  size_patch=(480, 360),
                  verbose=True,
                  configs_ransac=[{'min_samples': 0, 'residual_threshold': -1, 'max_trials': 0},

                                  {'min_samples': 4, 'residual_threshold': 1, 'max_trials': 100},
                                  {'min_samples': 4, 'residual_threshold': 1.5, 'max_trials': 100},
                                  {'min_samples': 4, 'residual_threshold': 2, 'max_trials': 100},
                                  {'min_samples': 4, 'residual_threshold': 2.5, 'max_trials': 100},
                                  {'min_samples': 4, 'residual_threshold': 3, 'max_trials': 100},
                                  ], threshold_inliers=10, random_select=True,angle_rotation = 0):
        path = self.name_model
        # Check whether the specified path exists or not
        isExist = os.path.exists(path)

        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(path)
            print("The new directory is created!")
        random.seed(9)
        count_yes = 0
        test_for_each_pair = 1

        paths = get_list()

        metrics = [{'rmse': 0, 'inliers': 0, 'accepted_match': 0} for conf in range(len(configs_ransac))]

        for ph in paths:
            for x in range(test_for_each_pair):


                results, entropy = do_test(self.make_inference, ph, size_search=size_search,
                                           size_patch=size_patch,
                                           verbose=verbose,
                                           configs_ransac=configs_ransac, threshold_inliers=threshold_inliers,
                                           random_select=random_select,
                                           name_model=self.name_model+"/"+pathlib.PurePath(ph[0]).parent.name+"_rotation_"+str(angle_rotation)+ "_"+ str(size_search)+"_"+str(size_patch),angle_rotation=angle_rotation)

                if results is not None:

                    for i in range(len(results)):
                        if results[i]['rmse'] >= 0:
                            metrics[i]['rmse'] += results[i]['rmse']
                            metrics[i]['inliers'] += results[i]['inliers']
                            metrics[i]['accepted_match'] += 1
                        print(results[i])
                        # print(metrics[i])

                    count_yes += 1

        data = []
        for i in range(len(metrics)):
            if metrics[i]['accepted_match'] > 0:
                metrics[i]['rmse'] /= metrics[i]['accepted_match']
                metrics[i]['inliers'] /= metrics[i]['accepted_match']

            configs_ransac[i]['rmse'] = metrics[i]['rmse']
            configs_ransac[i]['inliers'] = metrics[i]['inliers']
            configs_ransac[i]['Accepted_match'] = metrics[i]['accepted_match']
            configs_ransac[i]['total_match'] = test_for_each_pair * len(paths)
            data.append(configs_ransac[i])
            # print("Configuration:",configs_ransac[i-1]," Metrics:",metrics[i], " Accepted_match:", metrics[i]['accepted_match'] ,"/", test_for_each_pair*len(paths))

        import pandas as pd

        df = pd.DataFrame.from_dict(data)

        print(df)

        df.to_excel(self.name_model+"_rotation_"+str(angle_rotation)+ "_"+ str(size_search)+"_"+str(size_patch) +'_report.xlsx')
