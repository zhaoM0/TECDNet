import argparse

def get_option():
    parser = argparse.ArgumentParser(description="Training Parameter Setting")
    # model name
    parser.add_argument("--arch", type=str, required=False, default="RBF_TECDNet_S", help="model name")
    
    # dir setting 
    parser.add_argument("--pth_dir", type=str, default="./experiments/TECDNet-S-128", help="weights dir")
    parser.add_argument("--data_dir", type=str, default="E:/datasets/SIDD/SIDD_patches", help="data dir")
    parser.add_argument("--log_dir", type=str, default="./runs", help="log dir")

    # optimizer setting
    parser.add_argument("--optim", type=str, default="AdamW", help="optimizer")
    parser.add_argument("--is_warmup", type=bool, default=True, help="log dir")

    # data setting
    parser.add_argument("--augment", type=bool, default=True, help="is or not augment")
    parser.add_argument("--img_size", type=int, default=128, help="train image crops size")

    # training setting
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--n_epochs", type=int, default=250, help="training epochs")
    parser.add_argument("--lr_init", type=float, default=2e-4, help="initilize learning rate")
    parser.add_argument("--lr_min", type=float, default=1e-6, help="minimum learning rate")
    parser.add_argument("--n_warmup", type=int, default=5, help="starting round of warmup")
    parser.add_argument("--auto_save", type=int, default=20, help="auto saving epochs")
    # 
    parser.add_argument("--is_resume", type=bool, default=False, help="is or not resume")
    parser.add_argument("--device", type=str, default="cuda:0", help="calculate device")
    parser.add_argument("--resume_tag", type=str, default="best", help="resmue tag")

    args = parser.parse_args()
    return args 


def map_dict(args):
    assert type(args) == argparse.Namespace, "Input arguments error."
    args_dict = vars(args)
    return args_dict
