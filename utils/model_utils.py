import torch
import os
import glob
from collections import OrderedDict

def freeze(model):
    for p in model.parameters():
        p.requires_grad=False

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad=True

def is_frozen(model):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)

# save models
def save_models_v2(pth_dir, epoch, avg_psnr, model, model_name, optimizer, tag = 'best'):
    torch.save({
        'epoch': epoch,
        'avg_psnr': avg_psnr,
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optimizer.state_dict()
    }, os.path.join(pth_dir, f'{model_name}_{tag}.pth'))

def find_checkpoint(pth_dir, device, load_tag = 'best'):
    pth_zoo = "/*" + load_tag + "*.pth"
    special_pth = glob.glob(pth_dir + pth_zoo)
    if len(special_pth) != 0:
        checkpoint = torch.load(special_pth[0], map_location = device)
        return checkpoint
    else:
        print("[WARNING]: No pretrained weights in path `{}`.".format(pth_dir))
        return None

def show_checkpoint(checkpoint):
    if checkpoint is not None:
        print("Pretrained Model Message: ")
        print("1) Final saved epoch: %d." % (checkpoint['epoch']))
        print("2) Loss value: %2.7f." % (checkpoint['loss']))
        print("3) Best psnr value: %s." % (checkpoint['avg_psnr'].replace('-', '.')))
        print("4) Learning rate value: %0.7f." % (checkpoint['optim_state_dict']['param_groups'][0]['lr']))
    else:
        print("[WARNING]: Checkpoint file is none.")
