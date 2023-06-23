import gc
import os
import pathlib

import cv2
from typing import Tuple
import numpy as np
import pandas as pd
import rasterio

from torch import Tensor

import torch.onnx.utils

from matplotlib import pyplot as plt
from tqdm import tqdm

from Tester.tester import get_list, rotate_llh_map, do_test, do_pyramidal_test



class Tester():
    def __init__(self,matcher):
        self.name_model= "No_Name"
        self.matcher= matcher
        #self.optimizer = AdamW(self.matcher.parameters(), lr=0.001)

    def make_inference(self, image0: np.numarray, image1: np.numarray) -> Tuple[np.numarray, np.numarray, np.numarray]:
        return (np.array([]), np.array([]), np.array([]))


    def train_batch_(self,inputs, labels):
        self.matcher.train()
        # Zero your gradients for every batch!
        #self.optimizer.zero_grad()

        # Make predictions for this batch
        outputs = self.matcher(inputs)
        # Compute the loss and its gradients
        loss = torch.nn.MSELoss()(outputs, labels)
        loss.backward()
        # Adjust learning weights
        #self.optimizer.step()


    def make_pyramidal_test(self, size_search=(800, 720),
                  size_patch=(480, 360),
                  verbose=True,
                  configs_ransac=[{'min_samples': 0, 'residual_threshold': -1, 'max_trials': 0},

                                  {'min_samples': 4, 'residual_threshold': 1, 'max_trials': 100},
                                  {'min_samples': 4, 'residual_threshold': 1.5, 'max_trials': 100},
                                  {'min_samples': 4, 'residual_threshold': 2, 'max_trials': 100},
                                  {'min_samples': 4, 'residual_threshold': 2.5, 'max_trials': 100},
                                  {'min_samples': 4, 'residual_threshold': 3, 'max_trials': 100},
                                  ], threshold_inliers=0, random_select=True,angle_rotation = 0):
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


                results = do_pyramidal_test(self.make_inference, ph,(880,880), size_search=size_search,
                                           size_patch=size_patch,
                                           verbose=verbose,
                                           configs_ransac=configs_ransac, threshold_inliers=threshold_inliers,
                                           random_select=random_select,
                                           name_model=self.name_model+"/"+pathlib.PurePath(ph[0]).parent.name+"_rotation_"+str(angle_rotation)+ "_"+ str(size_search)+"_"+str(size_patch),angle_rotation=angle_rotation)
                del results


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

    from PIL import Image

    def make_test(self, size_search=(800, 720),
                  size_patch=(480, 360),
                  verbose=False,
                  configs_ransac=[{'min_samples': 0, 'residual_threshold': -1, 'max_trials': 0},
                                  {'min_samples': 4, 'residual_threshold': 0.5, 'max_trials': 100},
                                  {'min_samples': 4, 'residual_threshold': 1, 'max_trials': 100},
                                  {'min_samples': 4, 'residual_threshold': 1.5, 'max_trials': 100},
                                  {'min_samples': 4, 'residual_threshold': 3, 'max_trials': 100},
                                  {'min_samples': 4, 'residual_threshold': 5, 'max_trials': 100}],
                  threshold_inliers=10, random_select=True, angle_rotation=0):

        path = self.name_model
        if not os.path.exists(path):
            os.makedirs(path)
            print("The new directory is created!")

        random.seed(9)
        paths = get_list()

        configs_ransac_len = len(configs_ransac)
        metrics = [{'rmse': 0, 'inliers': 0, 'accepted_match': 0} for _ in range(configs_ransac_len)]
        paths_len = len(paths)

        for ii,ph in enumerate(tqdm(paths)):
            path_s2, path_s1, path_llh_uavsar, path_jpg_uavsar = ph
            uavsar = cv2.imread(path_jpg_uavsar, cv2.IMREAD_GRAYSCALE)
            #aU = np.nan_to_num((rasterio.open(path_s1).read()[0, :, :]))
            #bU = np.nan_to_num((rasterio.open(path_s2).read()[0, :, :]))

            #a = Image.fromarray(normalize_sar(aU, 99)*255)
            #b = Image.fromarray(normalize_sar(bU, 99)*255)
            #c = Image.fromarray(uavsar)


            #a.show(title="a")
            #b.show(title="b")
            #c.show(title="c")
            #axarr[0] = plt.imshow(a)
            #axarr[1] = plt.imshow(b)
            #axarr[2] = plt.imshow(c)

            #assert False
            llh = np.load(path_llh_uavsar)
            min_lat, max_lat = np.min(llh[:, :, 0]), np.max(llh[:, :, 0])
            min_lon, max_lon = np.min(llh[:, :, 1]), np.max(llh[:, :, 1])

            llh, uavsar = rotate_llh_map(uavsar, llh, path_s1)
            min_lat, max_lat = np.min(llh[:, :, 0]), np.max(llh[:, :, 0])
            min_lon, max_lon = np.min(llh[:, :, 1]), np.max(llh[:, :, 1])

            s1_norm_log = rasterio.open(path_s1).read()[0, :, :]#normalize_sar(, 97)



            ropen= rasterio.open(path_s1)
            parameters = (uavsar, llh, ropen , (np.squeeze(s1_norm_log) * 255).astype(np.uint8))
            name_model = self.name_model + "/" + pathlib.PurePath(ph[0]).parent.name + "_rotation_" + str(
                angle_rotation) + "_" + str(size_search) + "_" + str(size_patch)

            results = do_test(self.make_inference, parameters,ii, size_search=size_search,
                              size_patch=size_patch, verbose=verbose, configs_ransac=configs_ransac,
                              threshold_inliers=threshold_inliers, random_select=random_select,
                              name_model=name_model, angle_rotation=angle_rotation)

            if results is not None:
                for i in range(configs_ransac_len):
                    res = results[i]
                    if res['rmse'] >= 0:
                        metrics[i]['rmse'] += res['rmse']
                        metrics[i]['inliers'] += res['inliers']
                        metrics[i]['accepted_match'] += 1
                    if verbose :
                        print(res)


            del uavsar, llh, s1_norm_log,ropen,parameters
            gc.collect()

        data = []

        for i in range(configs_ransac_len):
            metric = metrics[i]
            accepted_match = metric['accepted_match']
            if accepted_match > 0:
                metric['rmse'] /= accepted_match
                metric['inliers'] /= accepted_match
            configs_ransac[i].update(metric)
            configs_ransac[i]['total_match'] = paths_len
            data.append(configs_ransac[i])
        pd.set_option('display.max_columns', None)
        df = pd.DataFrame.from_dict(data)
        print(df)
        df.to_excel(self.name_model + "_rotation_" + str(angle_rotation) + "_" + str(size_search) + "_" + str(
            size_patch) + '_report.xlsx')

def resize(img1, img2):
    # Store original shapes
    img1 = img1[0,0,:,:]
    img2 = img2[0,0,:,:]
    original_shape1 = img1.shape[:2]
    original_shape2 = img2.shape[:2]

    # Resize the images to be the same size
    target_size = (max(original_shape1[1], original_shape2[1]), max(original_shape1[0], original_shape2[0]))
    img1_resized = cv2.resize(img1, target_size).reshape(1,1,target_size[0],target_size[1])
    img2_resized = cv2.resize(img2, target_size).reshape(1,1,target_size[0],target_size[1])
    return img1_resized,img2_resized,original_shape1 , original_shape2,target_size
def rescale(mkpts0,mkpts1, original_shape1 , original_shape2,target_size ):
    # Apply your matching algorithm
    # Now, we rescale the points back to the original size
    scale1 = np.array([original_shape1[1]/target_size[0], original_shape1[0]/target_size[1]])
    scale2 = np.array([original_shape2[1]/target_size[0], original_shape2[0]/target_size[1]])

    mkpts0_rescaled = mkpts0 * scale1
    mkpts1_rescaled = mkpts1 * scale2

    return mkpts0_rescaled, mkpts1_rescaled
class Tester_Pytorch_TopicFM(Tester):
        def __init__(self,matcher,device="cuda"):
            super().__init__(matcher)
            self.name_model= "TopicFM"
            self.matcher = matcher.to(device)#get_matcher()
            self.device=device
        def make_inference(self,image0:np.numarray,image1:np.numarray)->Tuple[np.numarray,np.numarray,np.numarray]:


            m1,m2,c,m_bits =   self.matcher.matcher(torch.from_numpy(image0).to(self.device).float(),torch.from_numpy(image1).to(self.device).float())

            return m1.cpu().numpy(),m2.cpu().numpy(),c.cpu().numpy()
class Tester_ONNX(Tester):
        def __init__(self,matcher):
            super().__init__(matcher)
            self.name_model= "TopicFM_ONNX"
            self.matcher = matcher#get_matcher()

        def make_inference(self,image0:np.numarray,image1:np.numarray)->Tuple[np.numarray,np.numarray,np.numarray]:

            mkpts0 ,mkpts1,mconf,mbids = self.matcher.run(None, {"image0": image0.astype(np.float32),"image1": image1.astype(np.float32)})
            return mkpts0 ,mkpts1,mconf
class Tester_Pytorch_MatchFormer(Tester):
        def __init__(self,matcher,device):
            super().__init__(matcher)
            self.name_model= "MarchFormer"
            self.matcher = matcher.to(device)#get_matcher()
            self.device=device
        def make_inference(self,image0:np.numarray,image1:np.numarray)->Tuple[np.numarray,np.numarray,np.numarray]:

            img1_resized,img2_resized,original_shape1 , original_shape2,target_size = resize(image0,image1)
            batch = {'image0': torch.from_numpy(img1_resized).to(self.device).float(), 'image1': torch.from_numpy(img2_resized).to(self.device).float()}
            self.matcher.matcher(batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()
            mkpts0_r,mkpts1_r = rescale(mkpts0 ,mkpts1,original_shape1 , original_shape2,target_size)
            return mkpts0_r ,mkpts1_r,mconf
class Tester_PyTorch_ASpanFormer(Tester):
        def __init__(self,matcher,device):
            super().__init__(matcher)
            self.name_model= "ASpanFormer"
            self.matcher = matcher.to(device)#get_matcher()
            self.device=device
        def make_inference(self,image0:np.numarray,image1:np.numarray)->Tuple[np.numarray,np.numarray,np.numarray]:
            img1_resized,img2_resized,original_shape1 , original_shape2,target_size = resize(image0,image1)
            batch = {'image0': torch.from_numpy( img1_resized).to(self.device).float(), 'image1': torch.from_numpy( img2_resized).to(self.device).float()}

            self.matcher.matcher(batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()
            mkpts0,mkpts1 = rescale(mkpts0 ,mkpts1,original_shape1 , original_shape2,target_size)
            return mkpts0 ,mkpts1,mconf

def matrix2correspondences(matrix):
    matrix = matrix.permute(0, 2, 3, 1) # equivalent to torch.swapaxes(matrix, 1, -1)

    batch, width, height, _ = matrix.shape

    # Create meshgrid of coordinates
    x = torch.arange(width)
    y = torch.arange(height)
    xv, yv = torch.meshgrid(x, y)

    # Repeat xv and yv for the batch size
    xv_batch = xv.repeat(batch, 1, 1)
    yv_batch = yv.repeat(batch, 1, 1)

    # Form lst0 and lst1 using reshaping
    lst0 = torch.stack((xv_batch, yv_batch), dim=-1).reshape(batch, width*height, 2)
    lst1 = matrix.reshape(batch, width*height, 2)

    return lst0, lst1

class Tester_PyTorch_Res(Tester):
        def __init__(self,matcher,device):
            super().__init__(matcher)
            self.name_model= "ResCustom"
            self.matcher = matcher.to(device)#get_matcher()
            self.device=device
        def make_inference(self,image0:np.numarray,image1:np.numarray)->Tuple[np.numarray,np.numarray,np.numarray]:
            self.matcher.eval()
            with torch.no_grad():
                x_reconst = self.matcher(torch.from_numpy(image0).to(self.device).float(),torch.from_numpy(image1).to(self.device).float())
            self.matcher.train()

            mkpts0, mkpts1 = matrix2correspondences(x_reconst)

            return mkpts0.squeeze().cpu().numpy().astype('float64'),mkpts1.squeeze().cpu().numpy().astype('float64'),np.full((len(mkpts1.squeeze())), 0.98).astype('float64')

class Tester_PyTorch_AdaMatcher(Tester):
        def __init__(self,matcher,device):
            super().__init__(matcher)
            self.name_model= "AdaMatcher"
            self.matcher = matcher.to(device)#get_matcher()
            self.device=device
        def make_inference(self,image0:np.numarray,image1:np.numarray)->Tuple[np.numarray,np.numarray,np.numarray]:
            image0 = np.concatenate([image0,image0,image0],axis=1)#torch.cat([img, img, img], dim=0)
            image1 = np.concatenate([image1,image1,image1],axis=1)
            batch = {'image0': torch.from_numpy(image0).to(self.device).float(), 'image1': torch.from_numpy(image1).to(self.device).float()}

            self.matcher.matcher(batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()

            return mkpts0 ,mkpts1,mconf


class Tester_Pytorch_LoFTR(Tester):
        def __init__(self,matcher,device):
            super().__init__(matcher)
            self.name_model= "LoFTR"
            self.matcher = matcher.to(device)#get_matcher()
            self.device=device
        def make_inference(self,image0:np.numarray,image1:np.numarray)->Tuple[np.numarray,np.numarray,np.numarray]:
            batch = {'image0': torch.from_numpy(image0).to(self.device).float(), 'image1': torch.from_numpy(image1).to(self.device).float()}
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