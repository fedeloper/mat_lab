import cv2
import numpy as np
import rasterio
from memory_profiler import profile
#from line_profiler_pycharm import profile

from pyproj import Transformer
from scipy.ndimage import rotate
from sklearn.model_selection import train_test_split
from torch import nn, autograd
from torch.utils.data import Dataset, DataLoader, Subset
import math

from torchvision.transforms import transforms
from tqdm import tqdm

from DGC_Net.model.net import DGCNet
from Tester.tester import get_list, rotate_llh_map, get_sample, get_metrics_differentiable, create_geotiff
from Tester.tester_interface import Tester_PyTorch_Res, matrix2correspondences
from TopicFM_to_train.get_matcher import get_matcher
from d2net.lib.model import D2Net
from flownet2.networks.FlowNetS import FlowNetS
from model_trained.resnet import ResNet_VAE



class UAVSARDataset(Dataset):
    def __init__(self, paths, size_search=(800, 800), size_patch=(400, 400)):
        self.paths = paths
        self.size_search = size_search
        self.size_patch = size_patch
        self.crs_transform_list = {}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        path_s2, path_s1, path_llh_uavsar, path_jpg_uavsar = self.paths[idx]
        uavsar = cv2.imread(path_jpg_uavsar, cv2.IMREAD_GRAYSCALE)
        llh = np.load(path_llh_uavsar)
        llh, uavsar = rotate_llh_map(uavsar, llh, path_s1)
        ropen = rasterio.open(path_s1)
        s1_norm_log = ropen.read(1)  # normalize_sar(, 97)
        patch, search, sh0, sh1,off_x,off_y = get_sample((uavsar, llh, ropen, (np.squeeze(s1_norm_log) * 255)), self.size_search, self.size_patch)

        # Convert to PyTorch tensors
        patch = torch.tensor(patch, dtype=torch.float)
        search = torch.tensor(search, dtype=torch.float)

        self.crs_transform_list[idx] = (ropen,sh0, sh1)

        del uavsar,s1_norm_log
        return patch, search, llh, idx,off_x,off_y

    def get_crs_transform(self, idx):

        src,sh0,sh1 = self.crs_transform_list[idx.item()]
        return src,sh0,sh1


import torch

import numpy as np




def rotate_point(center_x, center_y, angle, x, y):
    temp_x = x - center_x
    temp_y = y - center_y
    rotated_x = temp_x * torch.cos(torch.tensor(angle)) - temp_y * torch.sin(torch.tensor(angle))
    rotated_y = temp_x * torch.sin(torch.tensor(angle)) + temp_y * torch.cos(torch.tensor(angle))
    return rotated_x + center_x, rotated_y + center_y


def get_metrics_batch_new(crs_indices,mks0,mks1 , llh, size_search, size_patch,off_x,off_y,dataset, angle_rotation=0,device="cuda"):
    loss= None
    for i in range(len(crs_indices)):

        src,sh0,sh1 =dataset.get_crs_transform(crs_indices[i])
        RMSE = get_metrics_differentiable(mks0[i], mks1[i], src, llh[i], size_search, sh0, sh1,
                                          off_x[i], off_y[i],crs_indices[i].item() ,angle_rotation=angle_rotation,device=device)
        #print(RMSE.requires_grad)
        #print(math.sqrt(RMSE.item()))
        if loss is None:
            loss = RMSE/len(crs_indices)
        else:
            loss += RMSE / len(crs_indices)
        src.close()
        del src,sh0,sh1
    return loss

def preprocess_patch_search(patch, search):

    patch = patch/ torch.max(patch)
    search = search/torch.max(search)

    return patch,search
#@profile
def train_one_iter(data,device,optimizer,net,size_search,size_patch,dataset,ANGLE_ROTATION):
    # if old is None:
    #    old = data
    # data = old
    patch, search, llh, idxs, off_x, off_y = data
    patch, search = preprocess_patch_search(patch.to(torch.float), search.to(torch.float))

    patch = patch.to(device).detach()
    search = search.to(device).detach()

    if patch.isnan().any() or search.isnan().any() or patch.isinf().any() or search.isinf().any():
        print("nan in inputs")

    # Forward pass
    optimizer.zero_grad()
    #outputs = net(search.unsqueeze(1).cpu(), patch.unsqueeze(1).cpu())
    mks0, mks1, conf, mbids = net(search.unsqueeze(1), patch.unsqueeze(1))

    #print(mks0.shape,mks1.shape)
    #matrix = outputs  # Assuming `matrix` is a NumPy array
    import random

    #mks0, mks1 = matrix2correspondences(matrix)
    #elected = random.sample(range(mks0.shape[1]), int(mks0.shape[1] / 100.00))
    loss = get_metrics_batch_new(idxs, mks0.unsqueeze(0), mks1.unsqueeze(0), llh, size_search, size_patch, off_x, off_y,
                                 dataset=dataset,
                                 angle_rotation=ANGLE_ROTATION,device=device)
    print(math.sqrt(loss.item()))
    loss.backward()

    optimizer.step()
    out = loss.cpu().detach().item()
    del loss,mks0,mks1,patch,search,llh,idxs,off_x,off_y
    return out
    #num_loss += 1
    #running_loss += loss.item()
#@profile
#@profile
def test_one_iter(data,device,size_search,size_patch,dataset,net,ANGLE_ROTATION):
    patch, search, llh, idxs, off_x, off_y = data
    patch, search = preprocess_patch_search(patch.to(torch.float), search.to(torch.float))
    patch = patch.to(device).detach()
    search = search.to(device).detach()
    mks0,mks1,conf,mbids=net(search.unsqueeze(1), patch.unsqueeze(1))
    #outputs = net(search.unsqueeze(1), patch.unsqueeze(1))

    #mks0, mks1 = matrix2correspondences(outputs)
    loss = get_metrics_batch_new(idxs, mks0, mks1, llh, size_search, size_patch, off_x, off_y, dataset=dataset,
                                 angle_rotation=ANGLE_ROTATION)
    out = loss.cpu().detach().item()
    del patch,search,llh,idxs,off_x,off_y,conf,mbids,loss
    return out
    #num_loss += 1
    #running_loss += loss.item()
def epoch(num,dataloader_train,dataloader_val,device,optimizer,net,size_search,size_patch,dataset,ANGLE_ROTATION):
    running_loss = 0.0
    num_loss = 0
    print("Starting training epoch:", num)
    import psutil
    print(psutil.virtual_memory())
    print("max memory allocated cuda", torch.cuda.max_memory_allocated(device="cuda"))
    torch.cuda.empty_cache()
    for i, data in enumerate(tqdm(dataloader_train)):
        running_loss += train_one_iter(data, device, optimizer, net, size_search, size_patch, dataset, ANGLE_ROTATION)
        num_loss += 1
    print("Epoch:", num, " training loss:", math.sqrt(running_loss / num_loss))
    running_loss = 0.0
    num_loss = 0
    print("Starting testing epoch:", num)
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader_val, 0)):
            running_loss += test_one_iter(data, device, size_search, size_patch, dataset, net, ANGLE_ROTATION)
            num_loss += 1
    print("Epoch:", num, " validation loss:", math.sqrt(running_loss / num_loss))
    # tester_py = Tester_PyTorch_Res(net, "cuda")
    # tester_py.make_test(size_search=(800, 800), size_patch=(400, 400))
    del running_loss,num_loss
def uu():
    size_search = (800, 800)
    size_patch = (400, 400)
    ANGLE_ROTATION = 0

    dataset = UAVSARDataset(get_list())
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # dataset = UAVSARDataset(get_list())
    dataloader_train = DataLoader(train_dataset, batch_size=1, shuffle=False,num_workers=0)
    dataloader_val = DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=0)
    device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #net = ResNet_VAE().to(device)  # Your model
    #net = FlowNetS().to(device)
    #model = D2Net(model_file="../d2net/models/d2_tf.pth",use_cuda=True)
    net = get_matcher().to(device).train().matcher
    #net = DGCNet().to(device)
    criterion = nn.MSELoss()  # Your loss function
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0001)  # Your optimizer

    # Number of epochs to train for
    import time

    #old = None
    num_epochs = 100
    for e in range(num_epochs):
        epoch(e,dataloader_train, dataloader_val, device, optimizer, net, size_search, size_patch, dataset,
              ANGLE_ROTATION)

    print('Finished Training')
if __name__=="__main__":
    print("aasdasdsad")
    uu()

