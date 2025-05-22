import sys
from argparse import ArgumentParser
from e3pose.rot.training import training

parser = ArgumentParser()

# PATHS
parser.add_argument("training_im_dir", type=str, help="data directory for training images")
parser.add_argument("training_lab_dir", type=str, help="data directory for ground-truth training segmentation labels")
parser.add_argument("training_anno_csv", type=str, help="path to ground-truth training canonical rotations")
parser.add_argument("val_im_dir", type=str, help="data directory for validation images")
parser.add_argument("val_lab_dir", type=str, help="data directory for ground-truth validation segmentation labels")
parser.add_argument("val_anno_csv", type=str, help="path to ground-truth validation canonical rotations")
parser.add_argument("results_dir", type=str, help="data directory for training outputs")

# GENERAL
parser.add_argument("--image_size", type=int, dest="image_size", default=64, help="input volume dimension in voxels")

# AUGMENTATION
parser.add_argument("--rotation", type=int, dest="rotation_range", default=90, help="random rotation parameter for data augmentation")
parser.add_argument("--shift", type=int, dest="shift_range", default=5, help="random translation parameter for data augmentation")
parser.add_argument("--norm_perc", type=float, dest="norm_perc", default=.005, help="fraction of input volume intensities that will be normalized to (0,1)")

# ARCHITECTURE
parser.add_argument("--n_levels", type=int, dest="n_levels", default=4, help="number of levels in E3CNN encoder")
parser.add_argument("--kernel_size", type=int, dest="kernel_size", default=5, help="convolution kernel size")

# TRAINING
parser.add_argument("--batch_size", type=int, dest="batch_size", default=1, help="batch size")
parser.add_argument("--lr", type=float, dest="learning_rate", default=0.01, help="learning rate")
parser.add_argument("--weight_decay", type=float, dest="weight_decay", default=3e-5, help="weight decay")
parser.add_argument("--momentum", type=float, dest="momentum", default=0.99, help="momentum for SGD")
parser.add_argument("--n_epochs", type=int, dest="n_epochs", default=100000, help="number of training epochs")
parser.add_argument("--validate_every_n_epoch", type=int, dest="validate_every_n_epoch", default=1, help="number of training epochs between every validation")
parser.add_argument("--resume", action='store_true', dest="resume", help="resume training from previous run")

# PRINT ALL ARGUMENTS
print('\nScript name:',  sys.argv[0])
print('\nScript arguments:')
args = parser.parse_args()
for arg in vars(args):
    print(arg, getattr(args, arg))
print('')

training(**vars(args))
