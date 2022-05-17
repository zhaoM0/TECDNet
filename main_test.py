import os 
import sys
import argparse
from pip import main
import torch 
import numpy as np
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import scipy.io as sio
from tqdm import tqdm
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from models import define_network

# ================================== Test Validate Statistic Message ===================================== #

def set_model(arch_name, pth_path, device="cuda:0"):
    # 
    device = torch.device(device)
    model = define_network(arch_name, {})
    checkpoint = torch.load(pth_path, map_location=device)
    assert checkpoint is not None, "In test process, model weight file doesn't exist."
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device) 
    return model 


def set_data(data_path):
    # load valid dataset.
    noisy_srgb_block = sio.loadmat(os.path.join(data_path, "ValidationNoisyBlocksSrgb.mat"))
    clean_srgb_block = sio.loadmat(os.path.join(data_path, "ValidationGtBlocksSrgb.mat"))
    
    noisy_block = np.float32(np.array(noisy_srgb_block['ValidationNoisyBlocksSrgb'])) / 255.
    clean_block = np.float32(np.array(clean_srgb_block['ValidationGtBlocksSrgb'])) / 255.   
    
    return noisy_block, clean_block


def test_sidd(model, noisy_block, clean_block, device="cuda:0"):
    # test validate dataset
    psnr_list, simm_list = list(), list()
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(40)):
            btc_clean = clean_block[i, :, :, :, :]
            btc_noisy = torch.from_numpy(noisy_block[i, :, :, :, :]).permute(0, 3, 1, 2).to(device)
            btc_repair = model(btc_noisy)
            btc_repair = torch.clamp(btc_repair, 0, 1).cpu().detach().permute(0, 2, 3, 1)
            btc_repair = btc_repair.numpy()

            for idx in range(32):
                psnr_list.append(psnr_loss(btc_repair[idx], btc_clean[idx]))
                simm_list.append(ssim_loss(btc_repair[idx], btc_clean[idx], multichannel=True))
    
    val_psnr_np = np.array(psnr_list)
    val_simm_np = np.array(simm_list)

    print("Validate dataset average psnr: %2.3f, average simm: %2.5f." % (np.mean(val_psnr_np), np.mean(val_simm_np)))
    return val_psnr_np, val_simm_np


def main_test_sidd(arch_name, pth_path, data_path, device):
    model = set_model(arch_name, pth_path, device)
    noisy_block, clean_block = set_data(data_path)
    psnr_arr, simm_arr = test_sidd(model, noisy_block, clean_block, device)
    return psnr_arr, simm_arr

# ================================== Inference for single image ===================================== #
def main_inference(arch_name, pth_path, img_path, device):
    # 
    model = set_model(arch_name, pth_path, device)
    
    input_img = Image.open(img_path)
    if input_img is None:
        raise Exception("Test image load error.")
    input_img = torch.from_numpy(np.array(input_img).astype(np.float32) / 255.)

    model.eval()
    with torch.no_grad():
        # create infernce patch
        ingput_img = input_img.unsqueeze(0).permute(0, 3, 1, 2).to(device)            
        repair_img = model(ingput_img)
        repair_img = torch.clamp(repair_img[0], 0, 1).cpu().detach().permute(1, 2, 0)
        repair_img = repair_img.numpy()

    # transform data
    repair_img = np.clip(repair_img * 255, 0, 255).astype(np.uint8)

    plt.imshow(repair_img)
    plt.show()

    return repair_img
# ======================================================================================================== #

if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Test setting.")
    parser.add_argument("--arch", type=str, default="RBF_TECDNet_S", help="model name")
    parser.add_argument("--pth_path", type=str, default="./experiments/TECDNet-S/RBF_TECDNet_S_best.pth", help="weights")
    parser.add_argument("--data_path", type=str, default="E:\\datasets\\SIDD\\SIDD_patches\\val", help="SIDD validate set")
    parser.add_argument("--device", type=str, default="cuda:0", help="calculate device")
    args = parser.parse_args()

    main_test_sidd(args.arch, args.pth_path, args.data_path, args.device)