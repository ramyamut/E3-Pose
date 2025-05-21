import numpy as np
import torch
import torchio

from . import utils

class AugmentationModel:
    def __init__(self,
                 im_shape, # [H, W, D]
                 img_res,
                 flipping=True,
                 scaling_bounds=(0.8, 2.0),
                 rotation_bounds=15,
                 translation_bounds=False,
                 nonlin_std=3.,
                 nonlin_scale=.0625,
                 randomise_res=False,
                 max_res_iso=8.,
                 max_bias=.5,
                 noise_std=0.08,
                 norm_perc=0.02,
                 gamma=0.4):
        # define all transformations to be applied
        self.identity_transform = torchio.transforms.Lambda(lambda x: utils.identity_transform(x))
        self.random_affine = torchio.transforms.RandomAffine(
            scales=(scaling_bounds[0], scaling_bounds[1], scaling_bounds[0], scaling_bounds[1], scaling_bounds[0], scaling_bounds[1]),
            degrees=(rotation_bounds, rotation_bounds, rotation_bounds),
            translation=(translation_bounds, translation_bounds, translation_bounds)
        )
        self.random_scale = torchio.transforms.RandomAffine(
            scales=(scaling_bounds[0], scaling_bounds[1], scaling_bounds[0], scaling_bounds[1], scaling_bounds[0], scaling_bounds[1]),
            degrees=0,
            translation=0
        )
        self.random_elastic = torchio.transforms.RandomElasticDeformation(
            num_control_points=tuple([int(im_shape[i]*nonlin_scale) for i in range(len(im_shape))]),
            max_displacement=nonlin_std,
        )
        self.random_flip = torchio.transforms.RandomFlip()
        self.max_bias = max_bias
        self.random_bias_field = RandomBiasField(coefficients=self.max_bias)
        normalize_percentiles = norm_perc if isinstance(norm_perc, tuple) else (norm_perc, 1.-norm_perc)
        self.normalize_intensities = torchio.transforms.Lambda(lambda x: utils.normalize_perc(x, normalize_percentiles), types_to_apply=[torchio.INTENSITY])
        self.max_res_iso = np.array(utils.reformat_to_list(max_res_iso, length=3, dtype='float'))
        self.iso_downsampling_scale = np.max(self.max_res_iso / img_res)
        self.random_isotropic_LR = RandomIsotropicLR(lr_scale=self.iso_downsampling_scale)
        resolution_prob = 0.75 if randomise_res else 0.
        self.resolution_transform = self.random_isotropic_LR
        self.random_noise = RandomNoise(max_std=noise_std)
        self.random_gamma = RandomGamma(log_gamma=gamma)
        self.random_haste = RandomSpinHistoryArtifact()
        self.normalize_intensities_final = Normalize()
        
        # compose transformations
        self.spatial_deform_transforms = torchio.transforms.Compose([
            self.random_affine,
            self.random_elastic
        ])
        flip_prob = 1. if flipping else 0.
        self.spatial_transforms = torchio.transforms.Compose([
            self.normalize_intensities_final,
            torchio.transforms.OneOf({
                self.spatial_deform_transforms: 0.95,
                self.identity_transform: 0.05
            }),
            torchio.transforms.OneOf({
                self.random_flip: flip_prob,
                self.identity_transform: 1-flip_prob
            })
        ])
        self.intensity_transforms = torchio.transforms.Compose([
            self.random_bias_field,
            self.normalize_intensities,
            self.random_haste,
            torchio.transforms.OneOf({
                self.resolution_transform: resolution_prob,
                self.identity_transform: 1-resolution_prob,
            }),
            torchio.transforms.OneOf({
                self.random_noise: 0.9,
                self.identity_transform: 0.1
            }),
            self.normalize_intensities_final,
            self.random_gamma,
        ])
        self.all_transforms = torchio.transforms.Compose([
            torchio.transforms.CropOrPad(64, mask_name='label'),
            self.spatial_transforms,
            self.intensity_transforms,
            CropPad()
        ])
    
    def get_transform(self):
        return self.all_transforms
    
    def get_val_transform(self):
        return torchio.transforms.Compose([
            torchio.transforms.CropOrPad(64, mask_name='label'),
            self.normalize_intensities_final
        ])

class RandomIsotropicLR(torchio.transforms.Transform):
    """
    Downsamples volumes to a random lower resolution, then resamples back to the original resolution
    """
    def __init__(self, p=1, lr_scale=1.):
        super().__init__(p=p)
        self.lr_scale = lr_scale
    
    def apply_transform(self, subject):
        sampled_scale = np.random.uniform(low=1., high=self.lr_scale)
        orig_spacing = subject['image'].spacing
        target = subject['image'].spatial_shape, subject['image'].affine
        spacing = tuple([orig_spacing[i]*sampled_scale for i in range(3)])
        downsample = torchio.transforms.Resample(target=spacing, scalars_only=True)
        downsampled = downsample(subject['image'])
        upsample = torchio.transforms.Resample(target=target, scalars_only=True)
        upsampled = upsample(downsampled)
        subject['image'] = upsampled
        return subject

class RandomSpinHistoryArtifact(torchio.transforms.Transform):
    """
    Augments volumes with simulated spin-history artifacts from high-resolution anatomical 2D MRI slices
    """
    def __init__(self, p=1, sigma=None):
        super().__init__(p=p)
        self.sigma = sigma
    
    def apply_transform(self, subject):
        mask = subject['label']
        vol = subject['image']
        try:
            vol_artifact = utils.simulate_spin_history_artifact(vol.tensor, mask.tensor, sigma=self.sigma)
        except:
            vol_artifact = torch.clone(vol.tensor)
        subject['image'] = torchio.ScalarImage(tensor=vol_artifact, affine=subject['image'].affine)
        return subject

class CropPad(torchio.transforms.Transform):
    """
    Randomly crops and zero-pads volume borders
    """
    def __init__(self, p=1):
        super().__init__(p=p)
    
    def apply_transform(self, subject):
        cutoff = np.random.randint(8,16)
        params = [0]*3
        params[np.random.randint(3)] = cutoff
        params = tuple(params)
        crop_transform = torchio.transforms.Compose([torchio.transforms.Crop(params), torchio.transforms.Pad(params)])
        transformed = crop_transform(subject)
        return transformed

class Normalize(torchio.transforms.Transform):
    """
    Applies min-max normalization to volumes, scaling intensity values to [0,1]
    """
    def __init__(self, p=1):
        super().__init__(p=p)
    
    def apply_transform(self, subject):
        subject['image'] = torchio.ScalarImage(tensor=utils.normalize(subject['image'].tensor), affine=subject['image'].affine)
        return subject

class RandomBiasField(torchio.transforms.Transform):
    """
    Randomly augments volumes with simulated bias fields
    """
    def __init__(self, p=1, coefficients=0.5):
        super().__init__(p=p)
        self.coefficients = coefficients
        self.order = 3
    
    def apply_transform(self, subject):
        sampled_coefs = self.get_params()
        coefs = {'image': sampled_coefs}
        order = {'image': self.order}
        bias_transform = torchio.transforms.BiasField(coefficients=coefs, order=order)
        transformed = bias_transform(subject)
        return transformed
    
    def get_params(self):
        random_coefficients = []
        for x_order in range(0, self.order + 1):
            for y_order in range(0, self.order + 1 - x_order):
                for _ in range(0, self.order + 1 - (x_order + y_order)):
                    sample = np.random.uniform(low=0., high=self.coefficients)
                    random_coefficients.append(sample)
        return random_coefficients

class RandomGamma(torchio.transforms.Transform):
    """
    Randomly augments volumes with gamma corrections (i.e. brightness variations)
    """
    def __init__(self, p=1, log_gamma=0.4):
        super().__init__(p=p)
        self.log_gamma_range = (-log_gamma, 0)
    
    def apply_transform(self, subject):
        sampled_gamma = self.get_params()
        gamma = {'image': sampled_gamma}
        gamma_transform = torchio.transforms.Gamma(gamma=gamma)
        transformed = gamma_transform(subject)
        return transformed
    
    def get_params(self):
        gamma = np.exp(np.random.uniform(low=self.log_gamma_range[0], high=self.log_gamma_range[1]))
        return gamma

class RandomNoise(torchio.transforms.Transform):
    """
    Randomly augments volumes with Gaussian noise
    """
    def __init__(self, p=1, max_std=0.):
        super().__init__(p=p)
        self.max_std = max_std
    
    def apply_transform(self, subject):
        vol = subject["image"]
        std = np.random.uniform(low=0, high=self.max_std)
        noise = torch.normal(torch.zeros_like(vol.tensor), torch.ones_like(vol.tensor)*std)
        subject['image'] = torchio.ScalarImage(tensor=vol.tensor+noise, affine=subject['image'].affine)
        return subject
