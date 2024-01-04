import os
import sys 
import time 
import json

import torch
import torch.optim as optim
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from config import get_option, map_dict

from utils.metrics import torch_psnr
from dataset import get_train_loader, get_val_loader, get_train_multi_loader
from utils.model_utils import find_checkpoint, save_models_v2
from utils.dataset_utils import split_image_256_to_128, splice_image_128_to_256
from utils.dataset_utils import MixUp_AUG
from utils.scheduler import GradualWarmupScheduler
from loss import charbonnier_loss
from utils.logger_utils import Logger

sys.path.append("models")
from models import define_network

def eval_step(model, val_loader, val_len, criterion, device='cuda:0'):
    model.eval()
    total_loss, total_psnr = 0, 0
    # iteration
    for _, batch_data in enumerate(val_loader):
        batch_size = batch_data[0].size(0)
        noisy_patch, clean_patch = batch_data[0].to(device), batch_data[1].to(device)
        with torch.no_grad():
            # inference
            repair_patch = model(noisy_patch)
            batch_loss = criterion(repair_patch, clean_patch)
            # update eval loss
            total_loss += batch_loss * batch_size
            # calculate psnr and loss of eval dataset)
            repair_patch = torch.clamp(repair_patch, min = 0., max = 1.)
            target_patch = torch.clamp( clean_patch, min = 0., max = 1.)
            # calculate psnr
            psnr_vec = torch_psnr(repair_patch, target_patch)
            total_psnr += torch.sum(psnr_vec)
    #
    avg_psnr = total_psnr / val_len
    avg_loss = total_loss / val_len
    return avg_loss, avg_psnr


# Main training function.
def train_pipeline(cfg):

    # create saving folder
    if not os.path.exists(cfg['pth_dir']):
        os.makedirs(cfg['pth_dir'])

    # setting logger
    logger = Logger(cfg['pth_dir'] + '/' + cfg['arch'] + '.log', write_log_file=True)
    writer = SummaryWriter(log_dir = os.path.join(cfg['log_dir'], cfg['arch']))

    now_time = time.strftime("%Y-%m-%d, %H:%M:%S", time.localtime())
    logger.write(f"Training start time is: {now_time}.")
    logger.write("1). Initilization, define network and data loader, Waiting ....")

    # setting environment
    device = torch.device(cfg["device"])
    model = define_network(cfg['arch'], {})
    model.to(device)

    logger.write(f"\t[arch name]: {cfg['arch']}")

    # define optimizer and learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr_init'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['n_epochs']-cfg['n_warmup'], eta_min=cfg['lr_min'])
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=cfg['n_warmup'], after_scheduler=scheduler)

    # get training dataset loader and eval dataset loader
    train_loader, train_len = get_train_loader(data_dir=cfg['data_dir'], batch_size=cfg['batch_size'], 
                                               num_workers=8, shuffle=True, img_size=cfg['img_size'])
    val_loader, val_len = get_val_loader(data_dir=cfg['data_dir'], batch_size=cfg['batch_size'], 
                                         num_workers=2, shuffle=False, img_size=256)

    logger.write(f"\t[datasets]: train length: {train_len}, validate length: {val_len}.")

    # training parameters
    last_epoch = 0 
    best_psnr  = 0.

    # if resume, then load checkpoint file.
    if cfg['is_resume'] is True:
        checkpoint = find_checkpoint(pth_dir=cfg['pth_dir'], device=device, load_tag=cfg['resume_tag']) 
        model.load_state_dict(checkpoint['model_state_dict'], strict=False) 
        optimizer.load_state_dict(checkpoint['optim_state_dict'])

    # auto resume 
    for _ in range(0, last_epoch + 1):
        scheduler_warmup.step()

    # define loss function and metric
    criterion = charbonnier_loss

    # evaluate step
    verbose_step = len(train_loader) // cfg['n_eval_of_each_epoch']

    # data augment
    mixup_aug = MixUp_AUG()

    # training iteration 
    logger.write("2). Training iteration:")
    for epoch in range(last_epoch + 1, cfg['n_epochs'] + 1):
        logger.write("epoch: %3d, lr: %.7f." % (epoch, optimizer.param_groups[0]['lr']))
        logger.write("=====================================")
        
        now_train_loss, now_iter_size = 0, 0
        train_tbar = tqdm(train_loader)
        
        model.train()
        for n_count, batch_data in enumerate(train_tbar):

            batch_size = batch_data[0].size(0)
            noisy_patch, clean_patch = batch_data[0].to(device), batch_data[1].to(device)

            if epoch > 5:
                noisy_patch, clean_patch = mixup_aug.aug(noisy_patch, clean_patch)

            # forward and backward
            optimizer.zero_grad()
            repair_patch = model(noisy_patch)
            batch_loss = criterion(repair_patch, clean_patch)
            batch_loss.backward()
            optimizer.step()
            
            # calculate total loss and update tqdm message
            now_train_loss += batch_loss * batch_size
            now_iter_size  += batch_size
            train_tbar.set_postfix(loss = (now_train_loss/now_iter_size).item())

            # validate 
            if (n_count + 1) % verbose_step == 0:
                avg_loss, avg_psnr = eval_step(model, val_loader, val_len, criterion, cfg['device'])
                model.train()
                
                ver_step = n_count + (epoch - 1) * len(train_loader)
                writer.add_scalar('val/psnr', avg_psnr, ver_step)
                writer.add_scalar('val/loss', avg_loss, ver_step)
                logger.write(f"\neval average psnr val is: {avg_psnr}, average loss is: {avg_loss}, best psnr: {best_psnr}.")

                # update best result
                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    save_models_v2(cfg['pth_dir'], epoch, best_psnr, model, cfg['arch'], optimizer, tag='best')

            if n_count % 40 == 0:
                ver_step = n_count + (epoch - 1) * len(train_loader)
                writer.add_scalar('train/loss', (now_train_loss/now_iter_size).item(), ver_step)

        save_models_v2(cfg['pth_dir'], epoch, best_psnr, model, cfg['arch'], optimizer, tag='last')
        logger.write(f"Average loss in this epoch is: {now_train_loss/now_iter_size}.\n")

        # update learing rate scheduler
        scheduler_warmup.step()        

    # log time 
    now_time = time.strftime("%Y-%m-%d, %H:%M:%S", time.localtime())
    logger.write("End Time Is: %s." % now_time) 


if __name__ == "__main__":
    # get argment from command line.
    args = get_option()
    cfg = map_dict(args)
    
    cfg_json = json.dumps(cfg, indent=4)
    print(cfg_json)

    # training process 
    train_pipeline(cfg=cfg)