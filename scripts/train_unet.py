import sys
from argparse import ArgumentParser
from e3pose.seg.training import training

parser = ArgumentParser()

# ------------------------------------------------- General parameters -------------------------------------------------

# PATHS
parser.add_argument("train_image_dir", help="data directory for training images")
parser.add_argument("train_labels_dir", help="data directory for ground-truth training segmentation labels")
parser.add_argument("val_image_dir", help="data directory for validation images")
parser.add_argument("val_labels_dir", help="data directory for validation images")
parser.add_argument("model_dir", help="data directory for training outputs")

# ---------------------------------------------- Augmentation parameters -----------------------------------------------

# OUTPUT
parser.add_argument("--batchsize", type=int, dest="batchsize", default=4, help="batch size")

# SPATIAL
parser.add_argument("--flipping", action='store_true', dest="flipping", help="random volume flipping during training")
parser.add_argument("--crop_size", dest="crop_size", type=int, default=128, help="volume crop size in voxels during training")
parser.add_argument("--scaling_lower_bound", dest="scaling_lower_bound", type=float, default=0.5, help="random scaling lower bound parameter for data augmentation")
parser.add_argument("--scaling_upper_bound", dest="scaling_upper_bound", type=float, default=1.3, help="random scaling upper bound parameter for data augmentation")
parser.add_argument("--rotation", dest="rotation_bounds", type=float, default=180, help="random rotation parameter for data augmentation")
parser.add_argument("--translation", dest="translation_bounds", type=float, default=10, help="random translation parameter for data augmentation")

# OTHER AUGMENTATIONS
parser.add_argument("--randomise_res", action='store_true', dest="randomise_res", help="augment training volumes with randomly simulated low resolutions")
parser.add_argument("--img_res", type=float, dest="img_res", default=3., help="input volume resolution during training")
parser.add_argument("--max_res_iso", type=float, dest="max_res_iso", default=8., help="max voxel dimension (in mm) for low-resolution augmentations")
parser.add_argument("--max_bias", type=float, dest="max_bias", default=.5, help="max magnitude of coefficients for bias field simulation")
parser.add_argument("--noise_std", type=float, dest="noise_std", default=.03, help="standard dev. for Gaussian noise data augmentation")
parser.add_argument("--norm_perc", type=float, dest="norm_perc", default=.005, help="fraction of input volume intensities that will be normalized to (0,1)")
parser.add_argument("--gamma", type=float, dest="gamma", default=.8, help="max log value of paremeter for gamma augmentation")

# -------------------------------------------- UNet architecture parameters --------------------------------------------
parser.add_argument("--n_levels", type=int, dest="n_levels", default=4, help="number of levels in U-Net")
parser.add_argument("--unet_feat", type=int, dest="unet_feat_count", default=16, help="number of output features in first layer of U-Net")
parser.add_argument("--feat_mult", type=int, dest="feat_multiplier", default=2, help="multiplicative factor for number of features after each U-Net level")
parser.add_argument("--activation", type=str, dest="activation", default='elu', help="activation function in U-Net")
parser.add_argument("--n_output_channels", type=int, dest="n_output_channels", default=3, help="number of output channels in U-Net (number of classes + 1)")

# ------------------------------------------------- Training parameters ------------------------------------------------

# GENERAL
parser.add_argument("--lr", type=float, dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--weight_decay", type=float, dest="weight_decay", default=0., help="weight decay")
parser.add_argument("--class_weights", nargs="*", dest="class_weights", default=[], help="class weights for dice and cross-entropy losses")
parser.add_argument("--dice_weight", type=float, dest="dice_weight", default=1., help="weight of dice loss when summed with cross-entropy loss")
parser.add_argument("--n_epochs", type=int, dest="n_epochs", default=1000, help="number of training epochs")
parser.add_argument("--resume", action='store_true', dest="resume", help="resume training from previous run")

# PRINT ALL ARGUMENTS
print('\nScript name:',  sys.argv[0])
print('\nScript arguments:')
args = parser.parse_args()
for arg in vars(args):
    print(arg, getattr(args, arg))
print('')
training(**vars(args))
