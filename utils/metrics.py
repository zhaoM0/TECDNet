import torch 
import numpy as np 

def torch_psnr(tar_img, prd_img):
    im_dif = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
    mse = torch.mean(im_dif**2, dim = (1, 2, 3))
    psnr = 10 * torch.log10(1 / mse)
    return psnr 

def numpy_psnr(tar_img, prd_img):
    im_dif = tar_img.astype(np.int32) - tar_img.astype(np.int32)
    mse = np.mean(im_dif ** 2)
    psnr = 10 * (2 * np.log10(255) - np.log10(mse))
    return psnr 
