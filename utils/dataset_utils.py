import torch
from einops import rearrange

class MixUp_AUG:
    def __init__(self):
        self.dist = torch.distributions.beta.Beta(torch.tensor([0.6]), torch.tensor([0.6]))

    def aug(self, rgb_gt, rgb_noisy):
        bs = rgb_gt.size(0)
        indices = torch.randperm(bs)
        rgb_gt2 = rgb_gt[indices]
        rgb_noisy2 = rgb_noisy[indices]

        lam = self.dist.rsample((bs,1)).view(-1,1,1,1).cuda()

        rgb_gt    = lam * rgb_gt + (1-lam) * rgb_gt2
        rgb_noisy = lam * rgb_noisy + (1-lam) * rgb_noisy2

        return rgb_gt, rgb_noisy

# [NOTE] Flip or Roatation
class RandomAugment:
    def __init__(self) -> None:
        pass
    def transform0(self, torch_tensor):
        return torch_tensor
    def transform1(self, torch_tensor):
        return torch.rot90(torch_tensor, k=1, dims=[-1, -2])
    def transform2(self, torch_tensor):
        return torch.rot90(torch_tensor, k=2, dims=[-1, -2])
    def transform3(self, torch_tensor):
        return torch.rot90(torch_tensor, k=3, dims=[-1, -2])
    def transform4(self, torch_tensor):
        return torch_tensor.flip(-2)
    def transform5(self, torch_tensor):
        return torch.rot90(torch_tensor, k=1, dims=[-1, -2]).flip(-2)
    def transform6(self, torch_tensor):
        return torch.rot90(torch_tensor, k=2, dims=[-1, -2]).flip(-2)
    def transform7(self, torch_tensor):
        return torch.rot90(torch_tensor, k=3, dims=[-1, -2]).flip(-2)


# split batch tensor from 256 to 128.
def split_image_256_to_128(patch_tensor):
    """ patch tensor with shape (1, 3, 256, 256) to (4, 3, 256, 256) """
    patch_128 = rearrange(patch_tensor, "b c (hn hs) (wn ws) -> (b hn wn) c hs ws", hn = 2, wn = 2)
    return patch_128


# concat batch tensor from 128 to 256.
def splice_image_128_to_256(patch_tensor):
    """ patch tensor from (4, 3, 128, 128) to (1, 3, 256, 256) """
    patch_256 = rearrange(patch_tensor, "(b hn wn) c hs ws -> b c (hn hs) (wn ws)", hn = 2, wn = 2)
    return patch_256
