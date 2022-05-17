import os 
import sys

import numpy as np
import scipy.io as sio
from tqdm import tqdm
from PIL import Image

def mat_transform_png():
    # prepare data
    noisy_block = sio.loadmat(os.path.join("data", "BenchmarkNoisyBlocksSrgb.mat"))
    noisy_block_np = np.array(noisy_block["BenchmarkNoisyBlocksSrgb"])

    for i in range(40):
        for j in range(32):
            noise_img = np.array(noisy_block_np[i, j, :, :, :])
            pil_img = Image.fromarray(noise_img)
            pil_img.save("data/benchmark/recst_%d.png" % (i * 32 + j))

if __name__ == "__main__":
    mat_transform_png()
    

