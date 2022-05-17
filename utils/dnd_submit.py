import numpy as np
import scipy.io as sio
import os
import h5py
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image


from models.base_unet_arch import resunet_c32_s4_at1, dual_resunet_standard, simple_unet_x2, simple_unet_x1, resunet_without_identity
from models.uformer_x0_arch import swin_former_v1_x32_128, swin_former_v2_standard_128
from models.uformer_x2_arch import swin_former_v1_standard_stocpath
from models.uformer_x3_arch import swin_former_v1_c16_stocpath_cft, swin_former_v1_c32_stocpath_conv, swin_former_v1_c32_stocpath_leff
from models.base_unet_x1_arch import base_unet_use_attention, base_unet_use_conv
from models.new_meta_net_arch import swin_encoder_conv_decoder_net_x1, swin_encoder_conv_decoder_net_c48, swin_encoder_conv_decoder_net_c16
from models.new_meta_net_arch import SECDNet_c16_bottle_no_swin, FFormer_c16_bottle_no_swin
from models.sparse_meta_net_arch import noswin_half_former_and_half_conv_net
from models.gauss_attention_utype_arch import RBF_SECDNet_C16, RBF_SECDNet_C32

from models.rbf_tecdnet_arch import RBF_TECDNet_S, RBF_TECDNet_L

def load_nlf(info, img_id):
    nlf = {}
    nlf_h5 = info[info["nlf"][0][img_id]]
    nlf["a"] = nlf_h5["a"][0][0]
    nlf["b"] = nlf_h5["b"][0][0]
    return nlf


def load_sigma_srgb(info, img_id, bb):
    nlf_h5 = info[info["sigma_srgb"][0][img_id]]
    sigma = nlf_h5[0,bb]
    return sigma


def denoise_srgb(denoiser, data_folder, out_folder):
    '''
    Utility function for denoising all bounding boxes in all sRGB images of
    the DND dataset.

    denoiser      Function handle
                  It is called as Idenoised = denoiser(Inoisy, nlf) where Inoisy is the noisy image patch 
                  and nlf is a dictionary containing the  mean noise strength (nlf["sigma"])
    data_folder   Folder where the DND dataset resides
    out_folder    Folder where denoised output should be written to
    '''
    try:
        os.makedirs(out_folder)
    except:pass

    # PTH_DIR = 'experiments_x1\\swin_encoder_conv_decoder_net_c48_2021-12-02\\swin_encoder_conv_decoder_net_c48_best.pth'
    # model = swin_encoder_conv_decoder_net_c48().cuda()

    PTH_DIR = 'experiments\\RBF_TECDNet_S_MD\\RBF_TECDNet_S_md_last.pth'
    model = RBF_TECDNet_S().cuda()

    pth_file = torch.load(PTH_DIR)
    model.load_state_dict(pth_file['model_state_dict'], strict = False)

    print('model loaded\n')
    # load info
    infos = h5py.File(os.path.join(data_folder, 'info.mat'), 'r')
    info = infos['info']
    bb = info['boundingboxes']
    print('info loaded\n')
    # process data
    for i in range(50):
        filename = os.path.join(data_folder, 'images_srgb', '%04d.mat'%(i+1))
        img = h5py.File(filename, 'r')
        Inoisy = np.float32(np.array(img['InoisySRGB']).T)

        # bounding box
        ref = bb[0][i]
        boxes = np.array(info[ref]).T
        for k in range(20):
            idx = [int(boxes[k,0]-1),int(boxes[k,2]),int(boxes[k,1]-1),int(boxes[k,3])]
            Inoisy_crop = Inoisy[idx[0]:idx[1],idx[2]:idx[3],:].copy()

            # Inoisy_crop = (Inoisy_crop * 255).astype(np.uint8)
            # plt.imshow(Inoisy_crop)
            # plt.show()
            # Inoisy_crop = Image.fromarray(Inoisy_crop)
            # Inoisy_crop.save(os.path.join(out_folder, 'src_crops', '%04d_%02d.png'%(i+1,k+1)), format='PNG')

            H = Inoisy_crop.shape[0]
            W = Inoisy_crop.shape[1]
            nlf = load_nlf(info, i)
            nlf["sigma"] = load_sigma_srgb(info, i, k)
            Idenoised_crop = denoiser(Inoisy_crop, nlf, model)

            # save denoised data
            # Idenoised_crop = np.float32(Idenoised_crop)
            save_file = os.path.join(out_folder, '%04d_%02d.mat'%(i+1,k+1))
            sio.savemat(save_file, {'Idenoised_crop': Idenoised_crop})
            
            print('%s crop %d/%d' % (filename, k+1, 20))
        print('[%d/%d] %s done\n' % (i+1, 50, filename))


def bundle_submissions_srgb(submission_folder):
    '''
    Bundles submission data for sRGB denoising
    
    submission_folder Folder where denoised images reside

    Output is written to <submission_folder>/bundled/. Please submit
    the content of this folder.
    '''
    out_folder = os.path.join(submission_folder, "bundled/")
    try:
        os.mkdir(out_folder)
    except:pass
    israw = False
    eval_version="1.0"

    for i in range(50):
        Idenoised = np.zeros((20,), dtype=np.object)
        for bb in range(20):
            filename = '%04d_%02d.mat'%(i+1,bb+1)
            s = sio.loadmat(os.path.join(submission_folder,filename))
            Idenoised_crop = s["Idenoised_crop"]
            Idenoised[bb] = Idenoised_crop
        filename = '%04d.mat'%(i+1)
        sio.savemat(os.path.join(out_folder, filename),
                    {"Idenoised": Idenoised,
                     "israw": israw,
                     "eval_version": eval_version},
                    )


def bundle_submissions_srgb_x2(submission_folder):
    out_folder = os.path.join(submission_folder, "bundled/")
    try:
        os.mkdir(out_folder)
    except:pass
    israw = False
    eval_version="1.0"

    for i in range(50):
        Idenoised = np.zeros((20,), dtype=np.object)
        for bb in range(20):
            filename = 'E:\\datasets\\DND\\train\\target\\' + '%04d_%02d.png'%(i+1,bb+1)
            Idenoised_crop = Image.open(filename)
            Idenoised_crop = np.array(Idenoised_crop).astype(np.float32) / 255.
            Idenoised[bb] = Idenoised_crop
        filename = '%04d.mat'%(i+1)
        sio.savemat(os.path.join(out_folder, filename),
                    {"Idenoised": Idenoised,
                     "israw": israw,
                     "eval_version": eval_version},
                    )


def inference(Inoisy, nlf, model):
    noisy = torch.from_numpy(Inoisy)
    noisy = noisy.unsqueeze(0).permute(0, 3, 1, 2).cuda()

    model.eval()
    with torch.no_grad():
        denoised = model(noisy)
        denoised = torch.clamp(denoised[0], 0., 1.).cpu().detach().permute(1, 2, 0)
        denoised = denoised.numpy()

    plt.subplot(1, 2, 1)
    plt.imshow(Inoisy)
    plt.subplot(1, 2, 2)
    plt.imshow(denoised)
    plt.show()

    return denoised


if __name__ == "__main__": 

    VALIDATE_DIR = ""
    OUT_DIR = "submit_dnd"

    denoise_srgb(inference, VALIDATE_DIR, OUT_DIR)
    bundle_submissions_srgb(OUT_DIR)
