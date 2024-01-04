## Hybrid Transformer-CNN for Real Image Denoising

#### Description

> We propose a hybrid denoising model based on Transformer Encoder and Convolutional Decoder
> Network (TECDNet), which can achieved great denosing performance while maintaining a relatively low computational complexity

#### Network Architecture

![](./fig/Fig-1.png)

#### Preparation

``` shell
$ conda create -n venv python=3.8
$ conda activate venv
$ pip install -r requirements.txt
```

#### Training

```
python main_train.py \
	--arch "RBF_TECDNet_S"  \
	--pth_dir "./experiments/TECDNet" \ 
	--data_dir "[your train data dir]" \ 
	--log_dir "./runs" \
	--is_warmup True \ 
	--augment True \
	--img_size 128 \
	--batch_size 32 \
	--n_epochs 250
```

#### Test

```
python main_test.py \
	--arch "RBF_TECDNet_S" \
	--pth_path "[the weights file dir]" \
	--data_path "[test images path]" \
	--device "cuda:0"
```

#### Models

Our pre-trained models can be downloaded as following:

|           | PSNR on SIDD | SSIM on SIDD | Weights Link |
| --------- | ------------ | ------------ | ------------ |
| TECDNet-S | 39.788       | 0.970        | [Link](https://drive.google.com/file/d/1gxal8QidK-KkG2K9YWSu3TmAo6ORoe9Y/view?usp=sharing)     |

#### Citation

```
@Article{9779501,
  author  = {Zhao, Mo and Cao, Gang and Huang, Xianglin and Yang, Lifang},
  journal = {IEEE Signal Processing Letters}, 
  title   = {Hybrid Transformer-CNN for Real Image Denoising}, 
  year    = {2022},
  volume  = {29},
  pages   = {1252-1256},
  doi     = {10.1109/LSP.2022.3176486}
}
```

#### Contact

If you have any questions, please contact me(koblod@163.com).