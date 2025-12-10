import sys
from argparse import ArgumentParser
from e3pose.inference import inference

parser = ArgumentParser()

# ------------------------------------------------- paths -------------------------------------------------
parser.add_argument("input_image_dir", help="data directory for input images")
parser.add_argument("output_dir", help="data directory for estimated poses")
parser.add_argument("unet_path", help="path to segmentation unet model weights")
parser.add_argument("e3cnn_path", help="path to e3cnn model weights")

# -------------------------------------------- UNet architecture parameters --------------------------------------------
parser.add_argument("--unet_crop_size", type=int, dest="unet_crop_size", default=128, help="volume crop size in voxels for segmentation U-Net")
parser.add_argument("--unet_n_levels", type=int, dest="unet_n_levels", default=4, help="number of levels in U-Net")
parser.add_argument("--unet_feat", type=int, dest="unet_feat_count", default=16, help="number of output features in first layer of U-Net")
parser.add_argument("--unet_feat_mult", type=int, dest="unet_feat_mult", default=2, help="multiplicative factor for number of features after each U-Net level")
parser.add_argument("--unet_activation", type=str, dest="unet_activation", default='elu', help="activation function in U-Net")
parser.add_argument("--unet_n_output_channels", type=int, dest="unet_n_output_channels", default=3, help="number of output channels in U-Net (number of classes + 1)")

# -------------------------------------------- E3CNN architecture parameters ------------------------------------------------
parser.add_argument("--e3cnn_image_size", type=int, dest="e3cnn_image_size", default=64, help="input volume dimension to e3cnn in voxels")
parser.add_argument("--e3cnn_n_levels", type=int, dest="e3cnn_n_levels", default=4, help="number of levels in E3CNN encoder")
parser.add_argument("--e3cnn_kernel_size", type=int, dest="e3cnn_kernel_size", default=5, help="convolution kernel size")

# -------------------------------------------- output pose parameters ------------------------------------------------
parser.add_argument("--scanner_space", action="store_true", dest="scanner_space", help="if specified, assume input coordinate frame is the scanner coordinate frame rather than the volume coordinate frame")
parser.add_argument("--input_to_canonical", action="store_true", dest="input_to_canonical", default=4, help="if specified, compute transform from input volume to canonical coordinate frame; otherwise compute transform from canonical to input frame")

# PRINT ALL ARGUMENTS
print('\nScript name:',  sys.argv[0])
print('\nScript arguments:')
args = parser.parse_args()
for arg in vars(args):
    print(arg, getattr(args, arg))
print('')
inference(**vars(args))
