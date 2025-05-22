import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import itertools
import torch
from torch.nn.functional import grid_sample
import torchio
from pytorch3d.transforms import matrix_to_euler_angles
from skimage.measure import label

# ----------------------------------------------- I/O functions -----------------------------------------------

def get_image_label_paths(image_dir, label_dir):
    path_images = sorted(glob.glob(os.path.join(image_dir, "*.nii"))) + sorted(glob.glob(os.path.join(image_dir, "*.nii.gz")))
    path_labels = [os.path.join(label_dir, os.path.basename(p)) for p in path_images]
    for p in path_labels:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} does not exist!")
    return path_images, path_labels

def list_images_in_folder(path_dir, include_single_image=True, check_if_empty=True):
    """List all files with extension nii, nii.gz, mgz, or npz within a folder."""
    basename = os.path.basename(path_dir)
    if include_single_image & \
            (('.nii.gz' in basename) | ('.nii' in basename) | ('.mgz' in basename) | ('.npz' in basename)):
        assert os.path.isfile(path_dir), 'file %s does not exist' % path_dir
        list_images = [path_dir]
    else:
        if os.path.isdir(path_dir):
            list_images = sorted(glob.glob(os.path.join(path_dir, '*nii.gz')) +
                                 glob.glob(os.path.join(path_dir, '*nii')) +
                                 glob.glob(os.path.join(path_dir, '*.mgz')) +
                                 glob.glob(os.path.join(path_dir, '*.npz')))
        else:
            raise Exception('Folder does not exist: %s' % path_dir)
        if check_if_empty:
            assert len(list_images) > 0, 'no .nii, .nii.gz, .mgz or .npz image could be found in %s' % path_dir
    return list_images

def read_vol(filename, im_only=True):
    x = nib.load(filename)
    vol = x.get_fdata()
    if vol.ndim == 3:
        vol = add_axis(vol, -1)
    if im_only:
        return vol
    else:
        aff = x.affine
        header = x.header
        return vol, aff, header

def save_volume(volume, aff, header, path):
    """
    Save a volume.
    :param volume: volume to save
    :param aff: affine matrix of the volume to save. If aff is None, the volume is saved with an identity affine matrix.
    aff can also be set to 'FS', in which case the volume is saved with the affine matrix of FreeSurfer outputs.
    :param header: header of the volume to save. If None, the volume is saved with a blank header.
    :param path: path where to save the volume.
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if header is None:
        header = nib.Nifti1Header()
    if aff is None:
        aff = np.eye(4)
    nifty = nib.Nifti1Image(volume, aff, header)
    nib.save(nifty, path)

def build_subject_frame_dict_all(im_dir, lab_dir=None):
    """Build a dictionary of the form
    {'subjID_frameID': [(path_frame, path_mask)]} if lab_dir is given
    {'subjID_frameID': [[path_frame]]}  if lab_dir is None.
    Note that each key is associated with a list of length 1.
    The image and labels paths are gotten from folders containing files with the following format subjID_frameID.nii.gz
    n_max_frame_per_subject is the maximum number of frames to get for one subject.
    min_start_frame is the minimum frame id to consider (because the very first frames of the time series can be bad).
    """
    data_dict = {}
    n_frames_per_subject = {}
    path_images = list_images_in_folder(im_dir)
    path_labels = list_images_in_folder(lab_dir) if lab_dir is not None else [None] * len(path_images)
    assert len(path_images) == len(path_labels), 'not the same number of images and labels'
    for path_im, path_lab in zip(path_images, path_labels):
        subjID_frameID = os.path.basename(path_im).split('.')[0]
        if path_lab is not None:
            data_dict[subjID_frameID] = [(path_im, path_lab)]  # each entry of the dict is a list with one tuple
        else:
            data_dict[subjID_frameID] = [[path_im]]  # each entry of the dict is a list with a list of 1
    return data_dict

def build_xfm_dict(xfm_dir):
    """Build a dictionary of the form {'subjID_frameID': path_xfm}."""
    data_dict = {}
    path_xfms = sorted(glob.glob(os.path.join(xfm_dir, '*.npy')))
    for path_xfm in path_xfms:
        subjID_frameID = os.path.basename(path_xfm).split('.')[0]
        data_dict[subjID_frameID] = path_xfm  # each entry of the dict is a list with a list of 1
    return data_dict

# ----------------------------------------------- preprocessing functions -----------------------------------------------

def preprocess_seg(image, aff):
    transform = torchio.transforms.Compose([
                                        torchio.transforms.RescaleIntensity(),
                                        torchio.transforms.Resample(3.),
                                        torchio.transforms.CropOrPad(64),
                                        torchio.transforms.RescaleIntensity(),
                                    ])
    image = torchio.ScalarImage(tensor=torch.tensor(image).unsqueeze(0), affine=aff)
    image_transformed = transform(image)
    return image_transformed.tensor, image_transformed.affine

def preprocess_rot(files, normalise=True, return_aff=False):

    # read volume
    aff = None
    if isinstance(files, str):
        if return_aff:
            vol, aff, h = read_vol(files, im_only=False)  # [H, W, D, C]
        else:
            vol = read_vol(files)
    elif isinstance(files, list):
        vol = np.concatenate([read_vol(f) for f in files], axis=-1)
    else:
        vol = files

    if normalise:
        vol = np.clip(vol, 0, None)
        m = np.min(vol)
        M = np.max(vol)
        vol = (vol - m) / (M - m)


    if return_aff:
        return vol, aff  # [H, W, D, C]
    else:
        return vol

def preprocess_rot_final(vol, lab, scale=0.6, resize=[64,64,64]):
    
    vol = vol.squeeze()
    lab = lab.squeeze()
    
    # crop out padded sections
    center, minc, maxc = get_bbox(vol != 0)
    vol_rolled = vol[minc[0]:maxc[0]+1, minc[1]:maxc[1]+1, minc[2]:maxc[2]+1]
    lab_rolled = lab[minc[0]:maxc[0]+1, minc[1]:maxc[1]+1, minc[2]:maxc[2]+1]
    
    # scaling
    vol_rolled, lab_rolled = crop_around_brain_scale(vol_rolled, lab_rolled, scale=scale)
    
    # resize to 64
    shape = np.array(vol_rolled.shape).astype(np.float32)
    shape *= np.array(resize) / np.max(shape)
    shape = np.round(np.array(shape)).astype(np.int32)
    resize = torchio.transforms.Compose([torchio.Resize(tuple(shape)), torchio.transforms.RescaleIntensity()])
    vol_rolled = resize(torchio.ScalarImage(tensor=torch.tensor(vol_rolled).unsqueeze(0))).tensor.squeeze().unsqueeze(-1).numpy()
    lab_rolled = resize(torchio.LabelMap(tensor=torch.tensor(lab_rolled).unsqueeze(0))).tensor.squeeze().unsqueeze(-1).numpy()
    

    return vol_rolled, lab_rolled

def crop_around_brain_scale(image, label, scale=0.6):
    brain_label = label.copy()
    brain_label[brain_label > 1] = 0
    brain_center, brain_corner1, brain_corner2 = get_bbox(brain_label)
    brain_extent = brain_corner2 - brain_corner1
    shape = np.array([brain_extent.mean()/scale]*3)
    brain_corner1 = np.round(brain_center-shape/2).astype(np.int32)
    brain_corner2 = np.round(brain_center+shape/2).astype(np.int32)
    brain_corner1 = np.maximum(brain_corner1, 0)
    brain_corner2 = np.minimum(brain_corner2, brain_label.squeeze().shape)
    bbox = [brain_corner1, brain_corner2]
            
    cropped_image = image[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1], bbox[0][2]:bbox[1][2]]
    cropped_label = label[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1], bbox[0][2]:bbox[1][2]]
    
    return cropped_image, cropped_label

# ----------------------------------------------- reformatting functions -----------------------------------------------
def reformat_to_list(var, length=None, load_as_numpy=False, dtype=None):
    """This function takes a variable and reformat it into a list of desired
    length and type (int, float, bool, str).
    If variable is a string, and load_as_numpy is True, it will be loaded as a numpy array.
    If variable is None, this function returns None.
    :param var: a str, int, float, list, tuple, or numpy array
    :param length: (optional) if var is a single item, it will be replicated to a list of this length
    :param load_as_numpy: (optional) whether var is the path to a numpy array
    :param dtype: (optional) convert all item to this type. Can be 'int', 'float', 'bool', or 'str'
    :return: reformatted list
    """

    # convert to list
    if var is None:
        return None
    var = load_array_if_path(var, load_as_numpy=load_as_numpy)
    if isinstance(var, (int, float, np.int32, np.int64, np.float32, np.float64)):
        var = [var]
    elif isinstance(var, tuple):
        var = list(var)
    elif isinstance(var, np.ndarray):
        if var.shape == (1,):
            var = [var[0]]
        else:
            var = np.squeeze(var).tolist()
    elif isinstance(var, str):
        var = [var]
    elif isinstance(var, bool):
        var = [var]
    if isinstance(var, list):
        if length is not None:
            if len(var) == 1:
                var = var * length
            elif len(var) != length:
                raise ValueError('if var is a list/tuple/numpy array, it should be of length 1 or {0}, '
                                 'had {1}'.format(length, var))
    else:
        raise TypeError('var should be an int, float, tuple, list, numpy array, or path to numpy array')

    # convert items type
    if dtype is not None:
        if dtype == 'int':
            var = [int(v) for v in var]
        elif dtype == 'float':
            var = [float(v) for v in var]
        elif dtype == 'bool':
            var = [bool(v) for v in var]
        elif dtype == 'str':
            var = [str(v) for v in var]
        else:
            raise ValueError("dtype should be 'str', 'float', 'int', or 'bool'; had {}".format(dtype))
    return var

def load_array_if_path(var, load_as_numpy=True):
    """If var is a string and load_as_numpy is True, this function loads the array writen at the path indicated by var.
    Otherwise it simply returns var as it is."""
    if (isinstance(var, str)) & load_as_numpy:
        assert os.path.isfile(var), 'No such path: %s' % var
        var = np.load(var)
    return var

# ----------------------------------------------- augmentation functions -----------------------------------------------

def normalize(x):
    x = x.float()
    min = x.min()
    max = x.max()
    return (x - min)/(max - min + 1e-8)

def add_noise(x, max_std):
    std = np.random.uniform(low=0, high=max_std)
    noise = torch.normal(torch.zeros_like(x), torch.ones_like(x)*std)
    return x + noise

def normalize_perc(x, perc):
    x = x.float()
    low = torch.quantile(x, perc[0])
    high = torch.quantile(x, perc[1])
    clipped = torch.clip(x, low, high)
    return normalize(clipped)

def identity_transform(input):
    return input

def sample_t_val_edge(beta_param=0.2, neg_prob=0.2):
    """
    Samples location of spin-history artifact so that the probabiliy it appears near the object edge is high (this is a harder case for segmentation)
    """
    if np.random.uniform() < neg_prob:
        uniform_dist = torch.distributions.uniform.Uniform(0., 0.3)
        t = uniform_dist.sample().item()
        if np.random.uniform() < 0.5:
            t = 0.1-t
        else:
            t = 0.9 + t
    else:
        beta_dist = torch.distributions.Beta(beta_param, beta_param)
        t = beta_dist.sample().item() * 0.8 + 0.1
    return t

def sample_t_val_uniform():
    """
    uniform sampling of spin-history artifact location
    """
    uniform_dist = torch.distributions.uniform.Uniform(-0.2, 1.2)
    t = uniform_dist.sample().item()
    return t

def simulate_spin_history_artifact(img, label, sigma_range, alpha_range, sample_t_uniform=False):
    """
    utility function to simulate spin-history artifact
    """
    img_shape = list(img.shape)
    artifact = torch.zeros_like(img) # [1, H, W, D]
    
    theta0 = 5
    label = (label == 1).to(torch.int32)
    label_boundary = torch.nn.functional.max_pool2d(1 - label.float(), kernel_size=theta0, stride=1, padding=(theta0 - 1) // 2)
    label_boundary -= 1 - label
    
    if torch.sum(label_boundary) == 0.:
        return torch.clone(img)
    
    boundary_idx = torch.nonzero(label_boundary[0])
    grid = torch.meshgrid(torch.linspace(-1, 1, img_shape[1]), torch.linspace(-1, 1, img_shape[2]), torch.linspace(-1, 1, img_shape[3]))
    grid = torch.stack(list(grid), dim=-1).to(img.device) # [H, W, D, 3]
    
    boundary_pts = grid[boundary_idx[:,0], boundary_idx[:,1], boundary_idx[:,2],:] # [points, 3 (xyz)]
    
    random_boundary_idx = torch.randperm(len(boundary_idx))
    p1 = boundary_pts[random_boundary_idx[0]] # [3]
    dx = boundary_pts - p1.unsqueeze(0) # [points, 3]
    dists = torch.sum(dx**2, dim=1)
    max_dist_idx = torch.argmax(dists).item()
    p2 = boundary_pts[max_dist_idx]
    dx = p2 - p1
    boundary_plane_n = torch.nn.functional.normalize(dx, dim=0) * 5
    
    if sample_t_uniform:
        t = sample_t_val_edge()
    else:
        t = sample_t_val_uniform()
    
    p = t * p1 + (1. - t) * p2 # [3 s(xyz)]
    
    d = (p @ boundary_plane_n).item()
        
    plane_prod = grid.reshape(-1, 3) @ boundary_plane_n
    plane_prod = plane_prod.reshape(img_shape) # [1, H, W, D]
    plane_prod = plane_prod - d
    
    # simulate artifact
    sigma = np.random.uniform(low=sigma_range[0], high=sigma_range[1])
    alpha = np.random.uniform(low=alpha_range[0], high=alpha_range[1])
    artifact = alpha * 1/(sigma * np.sqrt(2 * np.pi)) * torch.exp(-0.5 * (plane_prod/sigma)**2)
    
    img_with_artifact = torch.clone(img)
    img_with_artifact = img_with_artifact * torch.exp(-artifact)
    
    return img_with_artifact

def create_transform(rx, ry, rz, tx, ty, tz, ordering='txyz', input_angle_unit='degrees'):
    """create transformation matrix for rotation (degrees) and translation (pixels)."""

    if torch.is_tensor(rx):

        if input_angle_unit == 'degrees':
            if len(rx.shape) == 1:
                rx, ry, rz = torch.split(torch.cat([rx, ry, rz], dim=0) / 180 * np.pi, 1, dim=0)
            else:
                rx, ry, rz = torch.split(torch.cat([rx, ry, rz], dim=1) / 180 * np.pi, 1, dim=1)

        one = torch.ones_like(rx)
        zero = torch.zeros_like(rx)

        Rx = torch.cat([torch.stack([one, zero, zero, zero], dim=-1),
                        torch.stack([zero, torch.cos(rx), -torch.sin(rx), zero], dim=-1),
                        torch.stack([zero, torch.sin(rx), torch.cos(rx), zero], dim=-1),
                        torch.stack([zero, zero, zero, one], dim=-1)],
                       dim=-2)

        Ry = torch.cat([torch.stack([torch.cos(ry), zero, torch.sin(ry), zero], dim=-1),
                        torch.stack([zero, one, zero, zero], dim=-1),
                        torch.stack([-torch.sin(ry), zero, torch.cos(ry), zero], dim=-1),
                        torch.stack([zero, zero, zero, one], dim=-1)],
                       dim=-2)

        Rz = torch.cat([torch.stack([torch.cos(rz), -torch.sin(rz), zero, zero], dim=-1),
                        torch.stack([torch.sin(rz), torch.cos(rz), zero, zero], dim=-1),
                        torch.stack([zero, zero, one, zero], dim=-1),
                        torch.stack([zero, zero, zero, one], dim=-1)],
                       dim=-2)

        T = torch.cat([torch.stack([one, zero, zero, tx], dim=-1),
                       torch.stack([zero, one, zero, ty], dim=-1),
                       torch.stack([zero, zero, one, tz], dim=-1),
                       torch.stack([zero, zero, zero, one], dim=-1)],
                      dim=-2)

    else:

        if input_angle_unit == 'degrees':
            rx, ry, rz = np.array([rx, ry, rz]) * np.pi / 180

        if len(rx.shape) == 0:
            rx, ry, rz, tx, ty, tz = add_axis(np.array([rx, ry, rz, tx, ty, tz]), -1)

        one = np.ones_like(rx)
        zero = np.zeros_like(rx)

        Rx = np.concatenate([np.stack([one, zero, zero, zero], axis=-1),
                             np.stack([zero, np.cos(rx), -np.sin(rx), zero], axis=-1),
                             np.stack([zero, np.sin(rx), np.cos(rx), zero], axis=-1),
                             np.stack([zero, zero, zero, one], axis=-1)],
                            axis=-2)

        Ry = np.concatenate([np.stack([np.cos(ry), zero, np.sin(ry), zero], axis=-1),
                             np.stack([zero, one, zero, zero], axis=-1),
                             np.stack([-np.sin(ry), zero, np.cos(ry), zero], axis=-1),
                             np.stack([zero, zero, zero, one], axis=-1)],
                            axis=-2)

        Rz = np.concatenate([np.stack([np.cos(rz), -np.sin(rz), zero, zero], axis=-1),
                             np.stack([np.sin(rz), np.cos(rz), zero, zero], axis=-1),
                             np.stack([zero, zero, one, zero], axis=-1),
                             np.stack([zero, zero, zero, one], axis=-1)],
                            axis=-2)

        T = np.concatenate([np.stack([one, zero, zero, tx], axis=-1),
                            np.stack([zero, one, zero, ty], axis=-1),
                            np.stack([zero, zero, one, tz], axis=-1),
                            np.stack([zero, zero, zero, one], axis=-1)],
                           axis=-2)

    # final transform
    if ordering == 'xyzt':
        transform_matrix = Rx @ Ry @ Rz @ T
    elif ordering == 'xzyt':
        transform_matrix = Rx @ Rz @ Ry @ T
    elif ordering == 'yxzt':
        transform_matrix = Ry @ Rx @ Rz @ T
    elif ordering == 'yzxt':
        transform_matrix = Ry @ Rz @ Rx @ T
    elif ordering == 'zxyt':
        transform_matrix = Rz @ Rx @ Ry @ T
    elif ordering == 'zyxt':
        transform_matrix = Rz @ Ry @ Rx @ T

    elif ordering == 'txyz':
        transform_matrix = T @ Rx @ Ry @ Rz
    elif ordering == 'txzy':
        transform_matrix = T @ Rx @ Rz @ Ry
    elif ordering == 'tyxz':
        transform_matrix = T @ Ry @ Rx @ Rz
    elif ordering == 'tyzx':
        transform_matrix = T @ Ry @ Rz @ Rx
    elif ordering == 'tzxy':
        transform_matrix = T @ Rz @ Rx @ Ry
    elif ordering == 'tzyx':
        transform_matrix = T @ Rz @ Ry @ Rx
    else:
        raise ValueError('ordering should be a combination of the letters x,y,z with pre/appended t, got %s' % ordering)

    return transform_matrix

def aff_to_field(affine_matrix, image_size, invert_affine=False, rescale_values=False, keep_centred=False, centering=True):
    """Build a deformation field out of a transformation matrix. Can handle inputs with/without batch dim.
    :param affine_matrix: torch tensor or numpy array of size [B, n_dims + 1, n_dims + 1] or [n_dims + 1, n_dims + 1].
    :param image_size: size of the images that we will deform with the returned field. This excludes batch or channel
    dimensions, so it must be [H, W] (2D case) or [W, H, D] (3D case).
    :param invert_affine: whether to invert the affine matrix before computing the field. Useful for pull transforms.
    :param rescale_values: whether to rescale all the values of the field between [-1, 1], where [-1, -1] would be the
    top left corner and [1, 1] the bottom right corner for a 2d square. (useful for torch grid_sampler)
    :param keep_centred: whether to keep the center of coordinates at the center of the field.
    returns: a tensor of shape [B, *image_size, n_dims]"""

    n_dims = len(image_size)
    includes_batch_dim = len(affine_matrix.shape) == 3

    # make sure affine_matrix is float32 tensor
    if not torch.is_tensor(affine_matrix):
        affine_matrix = torch.tensor(affine_matrix, dtype=torch.float32)
    if affine_matrix.dtype != torch.float32:
        affine_matrix = affine_matrix.to(dtype=torch.float32)

    # meshgrid of coordinates
    coords = torch.meshgrid(*[torch.arange(s, device=affine_matrix.device, dtype=torch.float32) for s in image_size])

    # shift to centre of image
    if centering:
        offset = [(image_size[f] - 1) / 2 for f in range(len(image_size))]
        coords = [coords[f] - offset[f] for f in range(len(image_size))]
        if rescale_values | (not keep_centred):
            offset = add_axis(torch.tensor(offset, device=affine_matrix.device), [0] * (len(affine_matrix.shape) - 1))

    # add an all-ones entry (for homogeneous coordinates) and reshape into a list of points
    coords = [torch.flatten(f) for f in coords]
    coords.append(torch.ones_like(coords[0]))
    coords = torch.transpose(torch.stack(coords, dim=1), 0, 1)  # n_dims + 1 x n_voxels
    if includes_batch_dim:
        coords = add_axis(coords)

    # compute transform of each point
    if invert_affine:
        affine_matrix = torch.linalg.inv(affine_matrix)
    field = torch.matmul(affine_matrix, coords)  # n_dims + 1 x n_voxels
    if includes_batch_dim:
        field = torch.transpose(field, 1, 2)[..., :n_dims]  # n_voxels x n_dims
    else:
        field = torch.transpose(field, 0, 1)[..., :n_dims]  # n_voxels x n_dims

    # rescale values in [-1, 1]
    if rescale_values:
        field /= offset

    if centering and not keep_centred:
        field += offset

    # reshape field to grid
    if includes_batch_dim:
        new_shape = [field.shape[0]] + list(image_size) + [n_dims]
    else:
        new_shape = list(image_size) + [n_dims]
    field = torch.reshape(field, new_shape)  # *volshape x n_dims

    return field

def interpolate(vol, loc, batch_size=0, method='linear', vol_dtype=torch.float32):
    """Perform interpolation of provided volume based on the given voxel locations.

    :param vol: volume to interpolate. torch tensor or numpy array of size [dim1, dim2, ..., channel] or
    [B, dim1, dim2, ..., channel].
    WARNING!! if there's a batch dimension, please specify it in corresponding parameter.
    :param loc: locations to interpolate from. torch tensor or numpy array of size [dim1, dim2, ..., n_dims] or
    [B, dim1, dim2, ..., n_dims].
    :param batch_size: batch size of the provided vol and loc. Put 0 if these two tensors don't have a batch dimension.
    :param method: either "nearest" or "linear"
    :param vol_dtype: dtype of vol if we need to convert it from numpy array to torch tensor.

    returns: a pytorch tensor with the same shape as vol, where, for nearest interpolation in 3d we have
    output[i, j, k] = vol[loc[i, j, k, 0], loc[i, j, k, 1], loc[i, j, k, 2]]
    """
    # convert to tensor
    if not torch.is_tensor(vol):
        vol = torch.tensor(vol, dtype=vol_dtype)
    if not torch.is_tensor(loc):
        loc = torch.tensor(loc, device=vol.device, dtype=torch.float32)

    # get dimensions
    vol_shape_all_dims = list(vol.shape)
    vol_shape = vol_shape_all_dims[1:-1] if batch_size > 0 else vol_shape_all_dims[:-1]
    n_dims = loc.shape[-1]
    n_channels = vol_shape_all_dims[-1]
    vol = torch.reshape(vol, [-1, n_channels])

    if method == 'nearest':

        # round location values
        round_loc = torch.round(loc).to(torch.int32)

        # clip location values to volume shape
        max_loc = torch.tensor(vol_shape, device=vol.device, dtype=torch.int32) - 1
        max_loc = add_axis(max_loc, [0] * len(round_loc.shape[:-1]))
        round_loc = torch.clamp(round_loc, torch.zeros_like(max_loc), max_loc)

        # get values
        indices = coords_to_indices(round_loc, vol_shape, n_channels, batch=batch_size)
        vol_interp = torch.gather(vol, 0, indices).reshape(vol_shape_all_dims)

    elif method == 'linear':

        # get lower locations of the cube
        loc0 = torch.floor(loc)

        # clip location values to volume shape
        max_loc = [ss - 1 for ss in vol_shape]
        clipped_loc = [torch.clamp(loc[..., d], 0, max_loc[d]) for d in range(n_dims)]
        loc0_list = [torch.clamp(loc0[..., d], 0, max_loc[d]) for d in range(n_dims)]

        # get other end of point cube
        loc1_list = [torch.clamp(loc0_list[d] + 1, 0, max_loc[d]) for d in range(n_dims)]
        locs = [[tens.to(torch.int32) for tens in loc0_list], [tens.to(torch.int32) for tens in loc1_list]]

        # compute distances between points and upper and lower points of the cube
        dist_loc1 = [loc1_list[d] - clipped_loc[d] for d in range(n_dims)]
        dist_loc0 = [1 - d for d in dist_loc1]
        weights_loc = [dist_loc1, dist_loc0]  # note reverse ordering since weights are inverse of distances

        # go through all the cube corners, indexed by a binary vector
        vol_interp = 0
        cube_pts = list(itertools.product([0, 1], repeat=n_dims))
        for c in cube_pts:

            # get locations for this cube point
            tmp_loc = torch.stack([locs[c[d]][d] for d in range(n_dims)], -1)

            # get values for this cube point
            tmp_indices = coords_to_indices(tmp_loc, vol_shape, n_channels, batch=batch_size)
            tmp_vol_interp = torch.gather(vol, 0, tmp_indices).reshape(vol_shape_all_dims)

            # get weights for this cube point: if c[d] is 0, weight = dist_loc1, else weight = dist_loc0
            tmp_weights = [weights_loc[c[d]][d] for d in range(n_dims)]
            tmp_weights = torch.prod(torch.stack(tmp_weights, -1), dim=-1, keepdim=True)

            # compute final weighted value for each cube corner
            vol_interp = vol_interp + tmp_weights * tmp_vol_interp

    else:
        raise ValueError('method should be nearest or linear, had %s' % method)
    return vol_interp

def add_axis(x, axis=0):
    """Add axis to a numpy array or pytorch tensor.
    :param x: input array/tensor
    :param axis: index of the new axis to add. Can also be a list of indices to add several axes at the same time."""
    func = torch.unsqueeze if torch.is_tensor(x) else np.expand_dims
    if not isinstance(axis, list):
        axis = [axis]
    for ax in axis:
        x = func(x, ax)
    return x

def coords_to_indices(coords, vol_shape, n_channels, batch=0):
    cum_prod_shape = np.flip(np.cumprod([1] + list(vol_shape[::-1])))[1:]
    cum_prod_shape = add_axis(torch.tensor(cum_prod_shape.tolist(), device=coords.device), [0] * len(vol_shape))
    if batch > 0:
        cum_prod_shape = add_axis(cum_prod_shape)
    indices = torch.sum(coords * cum_prod_shape, dim=-1)
    if batch > 0:
        batch_correction = torch.tensor(np.arange(batch) * np.prod(vol_shape), device=coords.device)
        indices += add_axis(batch_correction, [-1] * len(vol_shape))
    indices = add_axis(indices.flatten(), -1).repeat(1, n_channels)
    return indices

# --------------------------------------------------- losses/metrics ----------------------------------------------------
def dice_coef(target, pred, smooth=1e-7):
    b = pred.size(0)
    n_classes = pred.size(1)
    pflat = pred.reshape(b, n_classes, -1)
    tflat = target.reshape(b, n_classes, -1)
    intersection = torch.sum((pflat * tflat), dim=2)
    dice = (2.0 * intersection + smooth) / (torch.sum(pflat, dim=2) + torch.sum(tflat, dim=2) + smooth)
    return dice

def dice(target, pred, smooth=1e-7):
    '''
    Returns dice coef, loss
    '''
    coef = dice_coef(target, pred, smooth=smooth)
    loss = 1 - coef # [B, C]
    return coef, loss

def rotation_matrix_to_angle_loss(true, pred):
    diff1 = torch.matmul(true, torch.transpose(pred, 1, 2))
    diff2 = torch.matmul(pred, torch.transpose(true, 1, 2))
    err_R1 = torch.abs(matrix_to_euler_angles(diff1, 'XYZ')) * 180/np.pi
    err_R2 = torch.abs(matrix_to_euler_angles(diff2, 'XYZ')) * 180/np.pi
    err_R = (err_R1 + err_R2)/2
    return err_R.mean()

# --------------------------------------------------- postprocessing/inference ----------------------------------------------------
def get_center_of_segmentation(label):
    """
    param label: binary mask, np array of size [H, W, D]
    """
    segmentation = np.where(label==1)
    x_min = int(np.min(segmentation[0]))
    x_max = int(np.max(segmentation[0]))
    y_min = int(np.min(segmentation[1]))
    y_max = int(np.max(segmentation[1]))
    z_min = int(np.min(segmentation[2]))
    z_max = int(np.max(segmentation[2]))
    center = (np.array([x_min, y_min, z_min]) + np.array([x_max, y_max, z_max]))/2
    return center

def get_bbox(label):
    segmentation = np.where(label==1)
    x_min = int(np.min(segmentation[0]))
    x_max = int(np.max(segmentation[0]))
    y_min = int(np.min(segmentation[1]))
    y_max = int(np.max(segmentation[1]))
    z_min = int(np.min(segmentation[2]))
    z_max = int(np.max(segmentation[2]))
    min_corner = np.array([x_min, y_min, z_min])
    max_corner = np.array([x_max, y_max, z_max])
    center = (min_corner + max_corner)/2
    return center, min_corner, max_corner

def get_largest_cc(mask):
    mask_int = mask.astype(np.uint8)
    labels =label(mask_int, connectivity = 1)
    object_labels = list(np.unique(labels))
    object_labels.remove(0)

    # get the continuous object that has the largest volume
    max_obj = 0
    max_obj_vol = 0
    for obj_lab in object_labels:
        vol = np.sum(labels == obj_lab)
        if vol > max_obj_vol:
            max_obj_vol = vol
            max_obj = obj_lab

    # remove every other object from the mask
    if max_obj != 0:
        object_labels.remove(max_obj)
    for obj_lab in object_labels:
        idx = (labels == obj_lab)
        mask[idx] = 0
    return mask

def postprocess_segmentation(raw_pred, thresh=0.15):
    
    raw_pred = raw_pred[0].detach().cpu()
    # threshold posteriors
    raw_pred[0, raw_pred[0] <= thresh] = 0
    raw_pred[1, raw_pred[1] <= 0.8] = 0
    
    final_post = torch.softmax(raw_pred, dim=0)
    mask = torch.argmax(final_post, dim=0).to(torch.int32) # [H, W, D]
    mask = mask.cpu().numpy()
    processed_mask = get_largest_cc((mask==1))
    out = torch.tensor(processed_mask).unsqueeze(0)
    return out.detach().cpu()

def axes_to_rotation(xax, yax, zax):
    xfm1 = torch.stack([torch.eye(3)]*xax.shape[0], dim=0)
    xfm1[:,:3,0] = xax
    xfm1[:,:3,1] = yax
    xfm1[:,:3,2] = zax
    xfm2 = torch.clone(xfm1)
    U1, _, Vh1 = torch.linalg.svd(xfm1)
    U2, _, Vh2 = torch.linalg.svd(xfm2)
        
    xfm1 = torch.bmm(U1, Vh1)
    xfm2 = torch.bmm(U2, Vh2)
    det = torch.det(xfm1)
    xfm = []
    for d in range(xfm1.shape[0]):
        if det[d] > 0:
            xfm.append(xfm1[d])
        else:
            xfm.append(xfm2[d])
    xfm = torch.stack(xfm, dim=0).to(xax.device)
    return xfm

def postprocess_rotation(raw_pred):
    pred = raw_pred[:,:,0,0,0].reshape(1, -1, 3).detach()
    pred = torch.nn.functional.normalize(pred,dim=2)
    xfm = torch.linalg.inv(axes_to_rotation(pred[:,0], pred[:,1], pred[:,2]))
    return xfm.squeeze().detach().cpu().numpy()

def get_rot_from_aff(aff):
    rot = np.copy(aff[:3,:3])
    for j in range(3):
        rot[:,j] /= np.linalg.norm(rot[:,j])
    return rot