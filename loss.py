import torch
from torch import tensor 
import torch.nn as nn 
import torch.nn.functional as F

# loss function 
def charbonnier_loss(reconstruct, target, eps = 1e-6):
    diff = torch.add(reconstruct, -target)
    error = torch.sqrt(diff * diff + eps)
    return torch.mean(error)


def charbonnier_loss_x1(repair, target, eps = 1e-6):
    diff = repair - target
    error = torch.sqrt(diff * diff + eps)
    return torch.mean(error, dim = (1, 2, 3))


def L2_loss(reconstruct, target):
    mean_mse = nn.MSELoss(reduction = 'mean')
    return mean_mse(reconstruct, target)


def var_correct_loss(repair, target, L_margin):
    diff  = repair - target
    error = torch.sqrt(diff * diff + 1e-6)
    loss_rt = torch.mean(error, dim = (1, 2, 3))
    loss_var = torch.clamp(loss_rt - L_margin, min = 0)
    loss_var = torch.mean(loss_var)
    return loss_var


def st_dist_loss(repair, target, v_val = 1.):
    diff = repair - target 
    core = 1 + (diff * diff) / v_val
    loss_tn = torch.log(core)
    return torch.mean(loss_tn)


def exp_aug_loss(repair, target, eps = 1e-6):
    diff = repair - target
    error = torch.sqrt(diff * diff + eps)
    error = torch.exp(error)
    return torch.mean(error)


if __name__ == "__main__":
    x = torch.rand((5, 3, 128, 128))
    y = torch.rand_like(x)

    print(st_dist_loss(x, y))
