# python imports
import os
import numpy as np
import torch
import glob
import nibabel as nib
import warnings
from scipy.ndimage import center_of_mass

# project imports
from .seg import unet
from .rot import networks
from . import utils

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
warnings.filterwarnings("ignore")

def inference(input_image_dir,
             output_dir,
             unet_path,
             e3cnn_path,
             unet_n_levels=4,
             unet_feat_count=16,
             unet_feat_mult=2,
             unet_activation='elu',
             unet_n_output_channels=3,
             seg_thresh=0.9,
             e3cnn_image_size=64,
             e3cnn_n_levels=4,
             e3cnn_kernel_size=5,
             scanner_space=False,
             canonical_to_input=False):
    
    # create network and lightning module
    print('LOADING NETWORKS...')
    unet = load_unet(unet_path, unet_n_output_channels, unet_n_levels, unet_feat_count, unet_feat_mult, unet_activation)
    e3cnn = load_e3cnn(e3cnn_path, e3cnn_n_levels, e3cnn_kernel_size)
    
    image_paths = sorted(glob.glob(os.path.join(input_image_dir, "*.nii"))) + sorted(glob.glob(os.path.join(input_image_dir, "*.nii.gz")))
    os.makedirs(output_dir, exist_ok=True)
    
    for i, path in enumerate(image_paths):
        print(f'PROCESSING VOLUME {i+1} of {len(image_paths)}')
        
        image_nii = nib.load(path)
        image = image_nii.get_fdata()
        aff = image_nii.affine
        aff_rot = utils.get_rot_from_aff(aff)
        
        # TRANSLATION ESTIMATION
        unet_input, unet_aff = utils.preprocess_seg(image, aff)
        unet_input = unet_input.to(device).unsqueeze(0).to(torch.float32)
        unet_output = torch.softmax(unet(unet_input), dim=1)
        seg_pred = utils.postprocess_segmentation(raw_pred=torch.clone(unet_output), thresh=seg_thresh)
        com = list(center_of_mass(seg_pred.squeeze().numpy())) # in unet input space, not original image space
        com_scanner = (unet_aff @ np.array(com + [1]))[:3]
        com_img = (np.linalg.inv(aff) @ unet_aff @ np.array(com + [1]))[:3]
        
        # ROTATION ESTIMATION
        e3cnn_input, _ = utils.preprocess_rot_final(unet_input.detach().cpu().squeeze().numpy(), seg_pred.detach().cpu().squeeze().numpy(), scale=0.6, resize=e3cnn_image_size)
        e3cnn_input = torch.tensor(e3cnn_input).squeeze().unsqueeze(0).unsqueeze(0).to(torch.float32).to(device)
        e3cnn_output = e3cnn.forward(e3cnn_input)
        rot_img = utils.postprocess_rotation(torch.clone(e3cnn_output))
        rot_scanner = aff_rot @ rot_img @ aff_rot.T
        
        # FULL RIGID POSE
        if scanner_space:
            rot = rot_scanner
            com = com_scanner
        else:
            rot = rot_img
            com = com_img
        pose_T = np.eye(4)
        pose_T[:3,3] = -com
        pose_R = np.eye(4)
        pose_R[:3,:3] = rot
        pose = pose_R @ pose_T
        
        if canonical_to_input:
            pose = np.linalg.inv(pose)
        
        # SAVE OUTPUT
        filename = os.path.basename(path).split(".nii")[0]
        output_path = os.path.join(output_dir, f"{filename}.npy")
        np.save(output_path, pose)


def load_unet(path_model, n_output_channels, n_levels, n_feat, feat_mult, activation):
    net = unet.UNet(
        n_input_channels=1,
        n_output_channels=n_output_channels,
        n_levels=n_levels,
        n_conv=2,
        n_feat=n_feat,
        feat_mult=feat_mult,
        kernel_size=3,
        activation=activation,
        last_activation=None
    ).to(device)
    state_dict = torch.load(path_model, map_location=device)["state_dict"]
    keys = list(state_dict.keys())
    new_state_dict = {}
    for k in keys:
        if 'model' in k:
            new_state_dict[k.replace('model.', '')] = state_dict[k]
    net.load_state_dict(new_state_dict, strict=True)
    net.eval()
    return net

def load_e3cnn(path_model, n_levels, kernel_size):
    net = networks.E3CNN_Encoder(input_chans=1, output_chans=1, n_levels=n_levels, k=kernel_size, last_activation=None, equivariance='O3')
    net = net.to(device)
    net.load_state_dict(torch.load(path_model, map_location=torch.device(device))['net_state_dict'])
    net.eval()
    return net
