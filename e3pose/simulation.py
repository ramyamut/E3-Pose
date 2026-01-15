import os
import glob
import pandas as pd
import nibabel as nib
import numpy as np
import torch
import torchio
import json
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interpn
from scipy.ndimage import center_of_mass
import time
from scipy.interpolate import interp1d

from e3pose import inference, utils, augmentation
from e3pose.rot.loaders import SpatialAugmenter
# from rot.loaders import SpatialAugmenter
# from release.src import augmenters, utils, networks
# from seg import augmentation, unet

def update_rigid_motion(rots, trans, frame, voxel_res):
    xfm_motion_R = np.eye(4)
    xfm_motion_R[:3,:3] = rots[frame]
    xfm_motion_T = np.eye(4)
    xfm_motion_T[:3,3] = trans[frame] / voxel_res
    xfm_motion = xfm_motion_R @ xfm_motion_T
    return xfm_motion

def compute_slice_aff_vol(slice_normal_canonical, slice_affine_canonical, rot, trans, slice_pos):
    slice_affine = np.eye(4)
    slice_normal = rot @ slice_normal_canonical
    slice_affine[:3,:3] = rot @ slice_affine_canonical
    slice_center = (trans + np.array(slice_normal) * slice_pos).tolist()
    return slice_affine, slice_center

def compute_slice_aff_canonical(slice_affine, xfm_tocanonical, slice_center, slice_center_vox, image_center, vol_aff):
    slice_center_canonical_vol = xfm_tocanonical[:3,:3] @ (slice_center-image_center) + image_center + xfm_tocanonical[:3,3]
    slice_center_scanner = (vol_aff @ np.array(slice_center_canonical_vol.tolist()+[1]))[:3]
    slice_affine_canonical = np.eye(4)
    slice_affine_canonical[:3,:3] = (vol_aff @ xfm_tocanonical @ slice_affine)[:3,:3]
    slice_affine_canonical[:3,3] = slice_center_scanner - (slice_affine_canonical[:3,:3] @ slice_center_vox)
    return slice_affine_canonical

def sample_motion_trajectory(sample_df,times):
        traj_idx = np.random.choice(pd.unique(sample_df['sample']))
        traj_df = sample_df[sample_df['sample']==traj_idx]
        traj_times = traj_df['t'].tolist()
        traj_tx = traj_df['tx'].tolist()
        traj_ty = traj_df['ty'].tolist()
        traj_tz = traj_df['tz'].tolist()
        traj_r = [Rotation.from_euler('xyz', [rx,ry,rz], degrees=True).as_matrix() for rx,ry,rz in zip(traj_df['rx'].tolist(), traj_df['ry'].tolist(), traj_df['rz'].tolist())]
        while traj_times[-1] < times[-1]:
                traj_idx = np.random.choice(pd.unique(sample_df['sample']))
                traj_df = sample_df[(sample_df['sample']==traj_idx) & (sample_df['t']>0)]
                traj_times += (traj_df['t']+traj_times[-1]).tolist()
                traj_tx += (traj_df['tx']+traj_tx[-1]).tolist()
                traj_ty += (traj_df['ty']+traj_ty[-1]).tolist()
                traj_tz += (traj_df['tz']+traj_tz[-1]).tolist()
                traj_r += [Rotation.from_euler('xyz', [rx,ry,rz], degrees=True).as_matrix()@traj_r[-1] for rx,ry,rz in zip(traj_df['rx'].tolist(), traj_df['ry'].tolist(), traj_df['rz'].tolist())]
        traj_slerp = Slerp(traj_times, Rotation.from_matrix(np.stack(traj_r, axis=0)))
        traj_interp = interp1d(np.array(traj_times), np.array([traj_tx,traj_ty,traj_tz]))
        traj_rots = traj_slerp(times)
        traj_trans = traj_interp(times).T
        return traj_rots.as_matrix(), traj_trans

def simulate(output_dir, image_dir, seg_label_dir, pose_label_csv, trajectory_csv, unet_path, e3cnn_path, acq_params, unet_params, e3cnn_params, run=0):
        list_images = sorted(glob.glob(f"{image_dir}/*nii.gz"))
        pose_label_df = pd.read_csv(pose_label_csv)
        trajectory_df = pd.read_csv(trajectory_csv)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        unet = inference.load_unet(unet_path, unet_params["n_output_channels"], unet_params["n_levels"], unet_params["n_feat"], unet_params["feat_mult"], unet_params["activation"])
        ecnn = inference.load_e3cnn(e3cnn_path, e3cnn_params["n_levels"], e3cnn_params["kernel_size"])
        
        n_interleave = acq_params["n_interleave"]
        slice_res = acq_params["slice_res"]
        slice_fov = acq_params["slice_fov"]
        slice_thickness = acq_params["slice_thickness"]
        navigator_res = acq_params["navigator_res"]
        dt=acq_params["dt"]
        tr=acq_params["tr"]
        
        slice_normals = {
                "axial": [0., 0., 1.],
                "sagittal": [-1., 0., 0.],
                "coronal": [0., 1., 0.]
        }
        slice_affines = {
                "axial": np.eye(3),
                "sagittal": np.array([
                        [0, 0, -1],
                        [-1, 0, 0],
                        [0, 1, 0]
                ]),
                "coronal": np.array([
                        [-1, 0, 0],
                        [0, 0, 1],
                        [0, 1, 0]
                ])
        }

        slice_center_vox = np.array([slice_fov/2, slice_fov/2, 0.5]) # slice center in slice voxel coordinates
        
        slice_scaling = np.eye(4)
        slice_scaling[:3,:3] = np.diag([slice_res, slice_res, slice_thickness])
        
        path_img = list_images[run]
        img_file = nib.load(path_img)
        image = img_file.get_fdata()
        aff = img_file.affine # affine of original image before cropping
        voxel_res = np.linalg.norm(aff[:,0]) # epi voxel dim (mm)
        vol_scaling = np.eye(4)
        vol_scaling[:3,:3] *= voxel_res
        slice_thickness_voxels = slice_thickness/voxel_res
        label_file = nib.load(path_img.replace(image_dir, seg_label_dir))
        label = (label_file.get_fdata()==1).astype(np.int32)
        filename = os.path.basename(path_img).split('.')[0]
        pose_label = pose_label_df[pose_label_df['filename'] == filename].iloc[0]
                
        xfm_canonical = np.eye(4)
        xfm_canonical[:3,:3] = Rotation.from_euler('xyz', [float(pose_label['rot_x']), float(pose_label['rot_y']), float(pose_label['rot_z'])], degrees=True).as_matrix()
        xfm_canonical = xfm_canonical.astype('float32')
                
        spatial_augmenter = SpatialAugmenter()
                
        shift_com = np.eye(4)
        shift_com[:3,3] = (np.array(label.squeeze().shape)-1)/2 - np.array(center_of_mass(label.squeeze()))
        xfm_canonical = xfm_canonical @ shift_com
        _, label_rot = spatial_augmenter.perform_transform(xfm_canonical, np.expand_dims(image, axis=-1), np.expand_dims(label, axis=-1))
        brain_center_canonical = np.array(center_of_mass(label_rot.squeeze()))
        xfm_canonical[:3,3] += (np.array(label_rot.squeeze().shape)-1)/2 - brain_center_canonical
        _, label_canonical = spatial_augmenter.perform_transform(xfm_canonical, np.expand_dims(image, axis=-1), np.expand_dims(label, axis=-1))
        
        run_dir = os.path.join(output_dir, filename)
        
        # SAMPLE TRAJECTORY
        _, b1, b2 = utils.get_bbox(label_canonical)
        box = aff[:3,:3] @ (b2-b1)
        for orientation_idx, orientation in enumerate(["axial", "coronal", "sagittal"]):
                n_slices_per_stack = int(np.ceil(box[orientation_idx]/slice_thickness))+4
                navigator_times = np.arange(n_slices_per_stack)*tr
                slice_times = navigator_times + dt
                times = np.stack([navigator_times,slice_times],axis=1).reshape(-1).tolist()
                traj_rots, traj_trans = sample_motion_trajectory(trajectory_df, times)
                navigator_rots = traj_rots[::2]
                navigator_trans = traj_trans[::2]
                slice_rots = traj_rots[1::2]
                slice_trans = traj_trans[1::2]
                
                stack_sampled_motion_dir = os.path.join(output_dir, "SAMPLED_MOTION", filename, orientation)
                for acq in ["navigator", "slice"]:
                        os.makedirs(os.path.join(stack_sampled_motion_dir, f"{acq}_pose"), exist_ok=True)
                xfm_navigator = np.eye(4)
                xfm_slice = np.eye(4)
                
                for frame in range(n_slices_per_stack):
                        xfm_slice = update_rigid_motion(slice_rots, slice_trans, frame, voxel_res)
                        
                        np.save(os.path.join(stack_sampled_motion_dir, "slice_pose", f"frame_{frame:03}.npy"), xfm_slice)
                        np.save(os.path.join(stack_sampled_motion_dir, "navigator_pose", f"frame_{frame:03}.npy"), xfm_navigator)
                        
                        if frame+1<n_slices_per_stack:
                                xfm_navigator = update_rigid_motion(navigator_rots, navigator_trans, frame+1, voxel_res)
        
        # RUN SIMULATION
        for orientation in ["axial", "sagittal", "coronal"]:
                stack_dir = os.path.join(run_dir, orientation)
                n_slices_per_stack = len(sorted(glob.glob(f"{output_dir}/SAMPLED_MOTION/{filename}/{orientation}/navigator_pose/*npy")))
                xfm_navigator = np.eye(4)
                xfm_slice = np.eye(4)
                        
                slice_zs_ordered = np.arange(0,n_slices_per_stack)*slice_thickness_voxels + slice_thickness_voxels/2
                slice_zs_ordered -= slice_zs_ordered.mean()
                slice_zs = []
                for sweep in range(n_interleave):
                        slice_zs += slice_zs_ordered[list(range(sweep, n_slices_per_stack, n_interleave))].tolist()
                
                slice_params_navigator = {}
                slice_params_slice = {}
                        
                fov = np.zeros((3,))
                        
                image_center = np.array([63]*3)
                brain_center_pred = np.copy(image_center)
                        
                for frame in range(n_slices_per_stack):
                        xfm_navigator = np.load(os.path.join(output_dir, "SAMPLED_MOTION", filename, orientation, "navigator_pose",  f"frame_{frame:03}.npy"))
                        dT = xfm_navigator @ np.linalg.inv(xfm_slice) # dt from slice to next navigator
                        xfm_slice = np.load(os.path.join(output_dir, "SAMPLED_MOTION", filename, orientation, "slice_pose",  f"frame_{frame:03}.npy"))
                        
                        xfm_fov_shift = np.eye(4)
                        xfm_fov_shift[:3,3] = -fov

                        image_label_motion = spatial_augmenter.perform_transform(xfm_fov_shift@xfm_navigator@xfm_canonical, np.expand_dims(image, axis=-1), np.expand_dims(label, axis=-1))
                        image_motion = image_label_motion[0].squeeze()
                        label_motion = image_label_motion[1].squeeze()
                        if frame == 0:
                                label_canonical = np.copy(label_motion)
                        
                        # augment image
                        subject = torchio.Subject(
                                image=torchio.ScalarImage(tensor=torch.tensor(image_motion).unsqueeze(0), affine=aff),
                                label=torchio.LabelMap(tensor=torch.tensor(label_motion).unsqueeze(0), affine=aff)
                        )
                        transform = torchio.transforms.Compose([
                                torchio.transforms.RescaleIntensity(),
                                torchio.transforms.CropOrPad(unet_params["crop_size"]),
                                augmentation.SpinHistoryArtifact(input_res=voxel_res, prescribe_params={"navigator": slice_params_navigator | acq_params, "slice": slice_params_slice | acq_params}, random=False),
                                torchio.transforms.RandomBlur((2.2,2.2)),
                                torchio.transforms.Resample(navigator_res),
                                torchio.transforms.Resample(voxel_res),
                                torchio.transforms.CropOrPad(unet_params["crop_size"]),
                                torchio.transforms.RescaleIntensity(),
                        ])
                        
                        transformed = transform(subject)
                        img_transformed = transformed['image']
                        label_transformed = transformed['label']
                        affine_transformed = img_transformed.affine
                        if frame == 0:
                                shift_aff_transformed = (aff@(((np.array(label_motion.squeeze().shape)-1)/2).tolist()+[1]) - affine_transformed@np.array(list(center_of_mass(label_transformed.tensor.squeeze().numpy()))+[1]))[:3]
                        affine_transformed[:3,3] += shift_aff_transformed
                        
                        # real-time inference
                        with torch.no_grad():
                                input_image = img_transformed.tensor.to(device).unsqueeze(0).to(torch.float32)
                                unet_pred = torch.softmax(unet(input_image), dim=1)

                                brain_seg_pred, eye_seg_pred = utils.postprocess_segmentation(raw_pred=torch.clone(unet_pred), thresh=[0.9,0.8,-1], fetal=True)
                                rot_pred_from_seg = utils.estimate_rot_from_seg(brain_seg_pred.squeeze().cpu().numpy(), eye_seg_pred.squeeze().cpu().numpy())
                                if brain_seg_pred.sum() == 0:
                                        brain_seg_pred = torch.ones_like(brain_seg_pred)
                                
                                brain_seg_gt = label_transformed.tensor.detach().cpu()
                                brain_center_pred = np.array(center_of_mass(brain_seg_pred.squeeze().numpy()))
                                brain_center_gt = np.array(center_of_mass(brain_seg_gt.squeeze().numpy()))

                                center_metric = brain_center_pred-brain_center_gt
                                
                                e3cnn_input, _ = utils.preprocess_rot_final(input_image.detach().cpu().squeeze().numpy(), brain_seg_pred.detach().cpu().squeeze().numpy())
                                e3cnn_input = torch.tensor(e3cnn_input).squeeze().unsqueeze(0).unsqueeze(0).to(torch.float32).to(device)
                                
                                ecnn_output = ecnn.forward(e3cnn_input)
                                rot_pred = utils.postprocess_rotation(torch.clone(ecnn_output))
                        
                        # update slice
                        slice_affine_pred, slice_center_pred = compute_slice_aff_vol(slice_normals[orientation], slice_affines[orientation], rot_pred, brain_center_pred, slice_zs[frame])
                        slice_affine_pred_from_seg, slice_center_pred_from_seg = compute_slice_aff_vol(slice_normals[orientation], slice_affines[orientation], rot_pred_from_seg, brain_center_pred, slice_zs[frame])
                        slice_affine_gt, slice_center_gt = compute_slice_aff_vol(slice_normals[orientation], slice_affines[orientation], xfm_navigator[:3,:3], brain_center_gt, slice_zs[frame])
                        
                        # calculate/display errors for debugging
                        err_R = Rotation.from_matrix(np.linalg.inv(xfm_navigator[:3,:3])@rot_pred).as_euler('xyz', degrees=True)
                        err_R_from_seg = Rotation.from_matrix(np.linalg.inv(xfm_navigator[:3,:3])@rot_pred_from_seg).as_euler('xyz', degrees=True)
                        print(f"frame {frame}: E3-Pose rot error {err_R}, seg-based rot error {err_R_from_seg}, trans error {center_metric}")
                        
                        # PRESCRIBE SLICE
                        fov_shift = brain_center_pred - image_center
                        slice_affine_canonical_pred = compute_slice_aff_canonical(slice_affine_pred@slice_scaling, np.linalg.inv(xfm_slice), slice_center_pred+fov, slice_center_vox, image_center, affine_transformed)
                        slice_affine_canonical_gt = compute_slice_aff_canonical(slice_affine_pred_from_seg@slice_scaling, np.linalg.inv(xfm_slice), slice_center_gt+fov, slice_center_vox, image_center, affine_transformed)
                        slice_affine_canonical_pred_from_seg = compute_slice_aff_canonical(slice_affine_gt@slice_scaling, np.linalg.inv(xfm_slice), slice_center_pred_from_seg+fov, slice_center_vox, image_center, affine_transformed)
                        slice_params_slice = {'orientation': (rot_pred@slice_normals[orientation]).tolist(), 'center': (slice_center_pred-fov_shift).tolist()}

                        # save navigator acquisitions and slice poses
                        navigator_dir = os.path.join(stack_dir, "navigators")
                        os.makedirs(navigator_dir, exist_ok=True)
                        nib.save(nib.Nifti1Image(img_transformed.tensor.squeeze().detach().cpu().numpy(), affine_transformed), os.path.join(navigator_dir, f"frame_{frame:03}.nii.gz"))
                        
                        if frame == 0:
                                nib.save(nib.Nifti1Image(label_canonical.squeeze(), aff), os.path.join(stack_dir, "mask.nii.gz"))
                        
                        slice_pred_dir = os.path.join(stack_dir, "slices_pred")
                        os.makedirs(slice_pred_dir, exist_ok=True)
                        nib.save(nib.Nifti1Image(np.ones((slice_fov, slice_fov, 1)), slice_affine_canonical_pred), os.path.join(slice_pred_dir, f"{frame}.nii.gz"))
                        nib.save(nib.Nifti1Image(np.ones((slice_fov, slice_fov, 1)), slice_affine_canonical_pred), os.path.join(slice_pred_dir, f"mask_{frame}.nii.gz"))
                        
                        slice_pred_from_seg_dir = os.path.join(stack_dir, "slices_pred_from_seg")
                        os.makedirs(slice_pred_from_seg_dir, exist_ok=True)
                        nib.save(nib.Nifti1Image(np.ones((slice_fov, slice_fov, 1)), slice_affine_canonical_pred_from_seg), os.path.join(slice_pred_from_seg_dir, f"{frame}.nii.gz"))
                        nib.save(nib.Nifti1Image(np.ones((slice_fov, slice_fov, 1)), slice_affine_canonical_pred_from_seg), os.path.join(slice_pred_from_seg_dir, f"mask_{frame}.nii.gz"))
                        
                        slice_gt_dir = os.path.join(stack_dir, "slices_gt")
                        os.makedirs(slice_gt_dir, exist_ok=True)
                        nib.save(nib.Nifti1Image(np.ones((slice_fov, slice_fov, 1)), slice_affine_canonical_gt), os.path.join(slice_gt_dir, f"{frame}.nii.gz"))
                        nib.save(nib.Nifti1Image(np.ones((slice_fov, slice_fov, 1)), slice_affine_canonical_gt), os.path.join(slice_gt_dir, f"mask_{frame}.nii.gz"))
                        
                        # update FOV
                        fov += fov_shift # FOV adjustment
                        
                        # SAMPLE MOTION FOR NEXT navigator
                        if frame+1<n_slices_per_stack:
                                slice_center_moved = (dT@((np.array(slice_center_pred)-fov_shift-image_center).tolist()+[1]))[:3]+image_center
                                slice_params_navigator = {'orientation': (dT[:3,:3]@rot_pred@slice_normals[orientation]).tolist(), 'center': slice_center_moved.tolist()}

def get_coverage_map_from_splatted(canonical_label, splatted):
        brain_coords_splatted = np.stack(np.where(np.ones(splatted.get_fdata().shape)),axis=-1)
        brain_coords_splatted = np.concatenate([brain_coords_splatted, np.ones((brain_coords_splatted.shape[0], 1))], axis=1).astype(np.int32)
        brain_coords_vol = (np.linalg.inv(canonical_label.affine) @ splatted.affine @ brain_coords_splatted.astype(np.float64).T).T[:,:3]
        valid_idx = np.logical_not(np.any(brain_coords_vol<0, axis=1))
        valid_idx = np.logical_and(valid_idx, brain_coords_vol[:,0]<=canonical_label.shape[0]-1)
        valid_idx = np.logical_and(valid_idx, brain_coords_vol[:,1]<=canonical_label.shape[1]-1)
        valid_idx = np.logical_and(valid_idx, brain_coords_vol[:,2]<=canonical_label.shape[2]-1)
        brain_coords_vol = brain_coords_vol[valid_idx]
        brain_coords_splatted = brain_coords_splatted[valid_idx]
        interp_values = interpn(
                tuple([np.arange(canonical_label.get_fdata().shape[i]) for i in range(3)]),
                canonical_label.get_fdata(),
                brain_coords_vol,
                method='nearest',
                fill_value=0
        )
        coverage_mask = np.zeros_like(splatted.get_fdata())
        coverage_mask[brain_coords_splatted[:,0], brain_coords_splatted[:,1], brain_coords_splatted[:,2]] = interp_values
        coverage_map_raw = np.copy(splatted.get_fdata())
        coverage_map_raw *= coverage_mask
        coverage_map = coverage_map_raw/np.sum(coverage_map_raw)
        return coverage_map, coverage_map_raw, coverage_mask

def generate_coverage_maps(data_dir, file_endings, simulation_idx=1, sub_idx=0):
        sub_dir = sorted(glob.glob(f"{data_dir}/*/"))[sub_idx]
        stack_dirs = sorted(glob.glob(f"{sub_dir}/*_{simulation_idx:03}/"))
        
        for stack_dir in stack_dirs:
                for file_ending in file_endings:
                        splatted_path = os.path.join(stack_dir, f"splatted{file_ending}.nii.gz")
                        canonical_label_path = os.path.join(stack_dir, "labels", "frame_000.nii.gz")
                        splatted = nib.load(splatted_path)
                        canonical_label = nib.load(canonical_label_path)
                        coverage_map, coverage_map_raw, coverage_mask = get_coverage_map_from_splatted(canonical_label, splatted)
                        nib.save(nib.Nifti1Image(coverage_map, splatted.affine), os.path.join(stack_dir, f"coverage_map{file_ending}.nii.gz"))
                        nib.save(nib.Nifti1Image(coverage_map_raw, splatted.affine), os.path.join(stack_dir, f"coverage_map_raw{file_ending}.nii.gz"))
                        nib.save(nib.Nifti1Image(coverage_mask, splatted.affine), os.path.join(stack_dir, f"coverage_mask{file_ending}.nii.gz"))
