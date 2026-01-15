import numpy as np
import warnings
import argparse
warnings.filterwarnings("ignore")

from e3pose.simulation import simulate

parser = argparse.ArgumentParser()

# --------------------------------------------- Main parameters -------------------------------------------------------
parser.add_argument("image_dir", type=str, help="directory of input volumes, each of which corresponds to a run")
parser.add_argument("seg_label_dir", type=str, help="directory containing ground-truth segmentation labels for volumes in image_dir")
parser.add_argument("pose_label_csv", type=str, help="path to CSV file containing ground-truth canonical rotations")
parser.add_argument("trajectory_csv", type=str, help="path to CSV file containing fetal head motion trajectories to sample from")
parser.add_argument("unet_path", type=str, help="path to segmentation unet model weights")
parser.add_argument("e3cnn_path", type=str, help="path to e3cnn model weights")
parser.add_argument("output_dir", type=str, help="output directory to save simulation results")
parser.add_argument("--run", type=int, dest="run", default=0, help="index ranging from 0 to N-1 (N = # of volumes in image_dir)")

# -------------------------------------------- Simulation parameters --------------------------------------------------
parser.add_argument("--n_interleave", type=int, dest="n_interleave", default=4, help="parameter for interleaved slice schedule")
parser.add_argument("--slice_res", type=float, dest="slice_res", default=1.0, help="in-plane slice resolution (mm)")
parser.add_argument("--slice_fov", type=int, dest="slice_fov", default=256, help="size of slice field-of-view in voxels")
parser.add_argument("--slice_thickness", type=float, dest="slice_thickness", default=3.0, help="slice thickness (mm)")
parser.add_argument("--navigator_res", type=float, dest="navigator_res", default=6.0, help="isotropic navigator voxel resolution (mm)")
parser.add_argument("--dt", type=float, dest="dt", default=1.0, help="time interval (s) between navigator and subsequent slice")
parser.add_argument("--tr", type=float, dest="tr", default=2.5, help="TR of the acquisition, i.e., time interval (s) between consecutive slices")

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

args = parser.parse_args()

acq_params = {
        "n_interleave": args.n_interleave,
        "slice_res": args.slice_res,
        "slice_fov": args.slice_fov,
        "slice_thickness": args.slice_thickness,
        "navigator_res": args.navigator_res,
        "dt": args.dt,
        "tr": args.tr
}
unet_params = {
      "crop_size": args.unet_crop_size,
      "n_levels": args.unet_n_levels,
      "n_feat": args.unet_feat_count,
      "feat_mult": args.unet_feat_mult,
      "activation": args.unet_activation,
      "n_output_channels": args.unet_n_output_channels,
}
e3cnn_params = {
      "image_size": args.e3cnn_image_size,
      "n_levels": args.e3cnn_n_levels,
      "kernel_size": args.e3cnn_kernel_size,
}

simulate(
        output_dir=args.output_dir,
        image_dir=args.image_dir,
        seg_label_dir=args.seg_label_dir,
        pose_label_csv=args.pose_label_csv,
        trajectory_csv=args.trajectory_csv,
        unet_path=args.unet_path,
        e3cnn_path=args.e3cnn_path,
        acq_params=acq_params,
        unet_params=unet_params,
        e3cnn_params=e3cnn_params,
        run=args.run,
)
