import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import torch

# ----------------------------------------------- I/O functions -----------------------------------------------

def get_image_label_paths(image_dir, label_dir):
    path_images = sorted(glob.glob(os.path.join(image_dir, "*.nii"))) + sorted(glob.glob(os.path.join(image_dir, "*.nii.gz")))
    path_labels = [os.path.join(label_dir, os.path.basename(p)) for p in path_images]
    for p in path_labels:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} does not exist!")
    return path_images, path_labels

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

def sample_t_val(beta_param=0.2, neg_prob=0.2):
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

def simulate_spin_history_artifact(img, label, sigma=None):
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
    
    t = sample_t_val()
    
    p = t * p1 + (1. - t) * p2 # [3 s(xyz)]
    
    d = (p @ boundary_plane_n).item()
        
    plane_prod = grid.reshape(-1, 3) @ boundary_plane_n
    plane_prod = plane_prod.reshape(img_shape) # [1, H, W, D]
    plane_prod = plane_prod - d
    
    # simulate artifact
    if sigma is not None:
        sigma = np.random.uniform(low=0.08, high=0.12)
    
    alpha = np.random.uniform(low=0.5, high=1.5)
    
    artifact = alpha * 1/(sigma * np.sqrt(2 * np.pi)) * torch.exp(-0.5 * (plane_prod/sigma)**2)
    
    img_with_artifact = torch.clone(img)
    img_with_artifact = img_with_artifact * torch.exp(-artifact)
    
    return img_with_artifact

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