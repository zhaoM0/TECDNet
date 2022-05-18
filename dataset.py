import os
import random
import numpy as np 
from PIL import Image
import scipy.io as sio
from glob import glob 

import torch
from torch.utils import data 
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from utils.dataset_utils import RandomAugment

def is_image_file(filename):
	return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG'])

# ======================================= Dataset Structure ======================================= #
# define dataset for denoise, 
# the dataset struture show here
# SIDD_Patches ——|—— train ——|—— input  ——|—— ... 
#                |           |—— traget ——|—— ... 
#                |——  val  ——|—— input  ——|—— ...
#                |           |—— target ——|—— ...
# 
# ======================================= Training Dataset ======================================== #
aug_obj = RandomAugment()
aug_transform = [method for method in dir(aug_obj) if not method.startswith('_')]

# Paired Image Datasets
class PairedImgDataset(Dataset):
    def __init__(self, root, tag = 'train', img_size = 128) -> None:
        super().__init__()
        self.img_size = img_size
        self.dir = os.path.join(root, tag)
        # create image list of noisy img and clean img 
        self.inp_files = sorted(os.listdir(os.path.join(self.dir, 'input')))
        self.tar_files = sorted(os.listdir(os.path.join(self.dir, 'target')))
        # 
        assert len(self.inp_files) == len(self.tar_files), "input numbers doesn't match target number."
        self.data_size = len(self.inp_files)

    def __getitem__(self, index):
        inp_nm, tar_nm = self.inp_files[index], self.tar_files[index]
        
        inp_img = Image.open(os.path.join(self.dir, 'input', inp_nm))
        tar_img = Image.open(os.path.join(self.dir, 'target', tar_nm))
        inp_img, tar_img = TF.to_tensor(inp_img), TF.to_tensor(tar_img)     # from [0, 255] to [0, 1]             

        img_h = inp_img.shape[1]
        img_w = inp_img.shape[2]

        if img_h - self.img_size == 0:
            r, c = 0, 0
        else:
            r = np.random.randint(0, img_h - self.img_size)
            c = np.random.randint(0, img_w - self.img_size)
        inp_img = inp_img[:, r : r + self.img_size, c : c + self.img_size]
        tar_img = tar_img[:, r : r + self.img_size, c : c + self.img_size]

        return inp_img, tar_img

    def __len__(self):
        return self.data_size

# ======================================= Multi Dataset Zoo ======================================== #
# Paired Image Datasets
class PairedImgMultiDataset(Dataset):
    def __init__(self, root_list, tag = 'train', img_size = 128) -> None:
        super().__init__()
        self.img_size = img_size
        # create image list of noisy img and clean img )
        self.inp_files = []
        self.tar_files = []

        for root in root_list:
            dir = os.path.join(root, tag)
            src_crops = sorted(glob(os.path.join(dir, 'input', '*')))
            tar_crops = sorted(glob(os.path.join(dir, 'target', '*')))
            self.inp_files = self.inp_files + src_crops
            self.tar_files = self.tar_files + tar_crops
        # 
        assert len(self.inp_files) == len(self.tar_files), "[ERROR] The dataset doesn't match."
        self.data_size = len(self.inp_files)

    def __getitem__(self, index):
        inp_nm, tar_nm = self.inp_files[index], self.tar_files[index]
        
        inp_img = Image.open(inp_nm)
        tar_img = Image.open(tar_nm)
        inp_img, tar_img = TF.to_tensor(inp_img), TF.to_tensor(tar_img)     # from [0, 255] to [0, 1]             

        img_h = inp_img.shape[1]
        img_w = inp_img.shape[2]

        if img_h - self.img_size == 0:
            r, c = 0, 0
        else:
            r = np.random.randint(0, img_h - self.img_size)
            c = np.random.randint(0, img_w - self.img_size)
        inp_img = inp_img[:, r : r + self.img_size, c : c + self.img_size]
        tar_img = tar_img[:, r : r + self.img_size, c : c + self.img_size]

        aug_trans = aug_transform[random.getrandbits(3)]
        inp_img = getattr(aug_obj, aug_trans)(inp_img)
        tar_img = getattr(aug_obj, aug_trans)(tar_img)

        return inp_img, tar_img

    def __len__(self):
        return self.data_size

# ================================================================================================== #

# define validate loader
def get_val_loader(data_dir, batch_size, num_workers, shuffle = False, img_size = 256):
    val_dataset = PairedImgDataset(data_dir, tag = 'val', img_size = img_size)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = shuffle, 
                            pin_memory = True, num_workers = num_workers)
    return val_loader, len(val_dataset)

# define train loader
def get_train_loader(data_dir, batch_size, num_workers, shuffle = True, img_size = 256):
    train_dataset = PairedImgDataset(data_dir, tag = 'train', img_size = img_size)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = shuffle,
                              pin_memory = True, num_workers = num_workers)
    return train_loader, len(train_dataset)


def get_train_multi_loader(data_dir, batch_size, num_workers, shuffle = True, img_size = 256):
    train_dataset = PairedImgMultiDataset(data_dir, tag = 'train', img_size = img_size)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = shuffle,
                              pin_memory = True, num_workers = num_workers)
    return train_loader, len(train_dataset)


# =========================================================================================== #
# dataset for bagging 
def sigmoid(arr):
    return 1 / (1 + np.exp(-arr))

class WeightedDataset(Dataset):
    def __init__(self, root, loss_arr, tag = 'train', img_size = 128, alpha = 1.) -> None:
        super().__init__()
        self.img_size = img_size
        self.dir = os.path.join(root, tag)
        
        # create image list of noisy img and clean img 
        inp_files = sorted(os.listdir(os.path.join(self.dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(self.dir, 'target')))
        
        # Core
        statistic = torch.mean(loss_arr)
        loss_norm = loss_arr - statistic
        threshold = sigmoid(alpha * loss_norm)
        uni_sample = np.random.uniform(low = 0, high = 1, size = len(inp_files))
        wei_sample = uni_sample < threshold

        # Sample rate 
        ori_sample = (loss_norm > 0).astype(np.float)
        print("Correct sample rate: %.3f." % (np.mean(ori_sample == wei_sample)))

        # Sample
        self.inp_files, self.tar_files = list(), list()
        for idx, select_element in enumerate(wei_sample):
            if select_element:
                self.inp_files.append(inp_files[idx])
                self.tar_files.append(tar_files[idx])
        # 
        assert len(self.inp_files) == len(self.tar_files), "input numbers doesn't match target number."
        self.data_size = len(self.inp_files)

    def __getitem__(self, index):
        inp_nm, tar_nm = self.inp_files[index], self.tar_files[index]
        
        inp_img = Image.open(os.path.join(self.dir, 'input', inp_nm))
        tar_img = Image.open(os.path.join(self.dir, 'target', tar_nm))
        inp_img, tar_img = TF.to_tensor(inp_img), TF.to_tensor(tar_img)     # from [0, 255] to [0, 1]             

        img_h = inp_img.shape[1]
        img_w = inp_img.shape[2]

        if img_h - self.img_size == 0:
            r, c = 0, 0
        else:
            r = np.random.randint(0, img_h - self.img_size)
            c = np.random.randint(0, img_w - self.img_size)
        inp_img = inp_img[:, r : r + self.img_size, c : c + self.img_size]
        tar_img = tar_img[:, r : r + self.img_size, c : c + self.img_size]

        return inp_img, tar_img

    def __len__(self):
        return self.data_size
