import copy
import torch
import numpy as np
import os
import torch.utils.data
import numpy.random as npr
import nibabel as nib
import torchio
from scipy.spatial.transform import Rotation

from .. import utils, augmentation

class loader_rot_canonical(torch.utils.data.IterableDataset):
    """dataloader class that takes in a list of scans & masks and a dataframe of ground-truth canonical rotations per scan
    """

    def __init__(self,
                 subj_dict,
                 anno_df,
                 augm_params=None,
                 eval=False,
                 seed=None):

        # input data
        self.subj_dict = subj_dict
        self.anno_df = anno_df
        self.list_of_subjects = list(self.subj_dict.keys())
        self.n_samples = len(self.list_of_subjects)
        self.iter_idx = 0
        self.resize = augm_params["resize"]

        # initialise resize/rescale functions
        self.preproc_func = lambda x: utils.preprocess_rot(x, normalise=True)
        self.preproc_func_labels = lambda x: utils.preprocess_rot(x, normalise=False)

        # numpy seed
        self.eval = eval
        self.rng = npr.RandomState(seed)
        
        img = nib.load(self.subj_dict[self.list_of_subjects[0]][0][0])
        img_res = np.array(img.header['pixdim'][1:4])

        # load, resize, rescale images/labels (still numpy with size [H, W, D, C])
        self.samples = {}  # {'subj_id': [[frame_1, mask_1], [fr_2, mask_2], ...]}
        for subj in self.list_of_subjects:
            self.samples[subj] = []
            for scan_name_tuple in self.subj_dict[subj]:
                self.samples[subj].append(self.load_sample(scan_name_tuple))

        # initialise spatial/intensity augmenters
        self.spatial_augmenter = SpatialAugmenter(rotation_range=augm_params["rotation_range"],
                                                    shift_range=augm_params["shift_range"],
                                                    return_affine=True,
                                                    normalise=True,
                                                    seed=seed)
        self.intensity_augmenter = IntensityAugmenter(augm_params, seed=seed)

        # output format params
        self.output_names = ["image", "mask", "rot"]

    def load_sample(self, sample_tuple):
        image = self.preproc_func(sample_tuple[0])  # [H, W, D, C]
        labels = self.preproc_func_labels(sample_tuple[1])

        return [image, labels]

    def __next__(self):

        if self.iter_idx >= self.n_samples:
            self.iter_idx = 0  # reset at the end of every epoch
            raise StopIteration

        # load a random [frame, mask] from a random subject
        idx = self.rng.choice(self.n_samples) if not self.eval else self.iter_idx % self.n_samples
        frame_id = np.random.choice(len(self.samples[self.list_of_subjects[idx]]))
        frame_mask_1 = copy.deepcopy(self.samples[self.list_of_subjects[idx]][frame_id])  # [frame, mask], [H, W, D, C]
        frame_mask_1[1] = np.round(frame_mask_1[1])
        sub_frame_id = self.list_of_subjects[idx]
        anno = self.anno_df[self.anno_df['frame_id'] == sub_frame_id].iloc[0]
        
        xfm_2 = np.eye(4)
        xfm_2[:3,:3] = Rotation.from_euler('xyz', [float(anno['rot_x']), float(anno['rot_y']), float(anno['rot_z'])], degrees=True).as_matrix()
        xfm_2 = xfm_2.astype('float32')

        # spatial augment, outputs numpy matrices in all cases
        frame_mask_1[1][frame_mask_1[1]>1] = 0
        frame_mask_1[1] = (frame_mask_1[1]==1).astype(np.int32)
        if self.eval:
            xfm_1 = np.eye(4)
        else:
            xfm_1 = self.spatial_augmenter.create_random_transform()
        xfm_1 = xfm_1.astype('float32') @ xfm_2
        frame_mask_1[0], frame_mask_1[1] = utils.preprocess_rot_final(frame_mask_1[0], frame_mask_1[1], scale=0.6, resize=self.resize)
        frame_mask_1, _ = self.spatial_augmenter.perform_transform(xfm_1, *frame_mask_1)
        xfm_1 = np.linalg.inv(xfm_2 @ np.linalg.inv(xfm_1.astype('float32')))

        # intensity augment
        label_for_augm = frame_mask_1[1].copy()
        label_for_augm[label_for_augm > 1] = 0
        
        if self.eval:
            frame_mask_1[0], frame_mask_1[1] = self.intensity_augmenter.eval_transform(frame_mask_1[0], label_for_augm)
        else:
            frame_mask_1[0], frame_mask_1[1] = self.intensity_augmenter.training_transform(frame_mask_1[0], label_for_augm)

        # group outputs in dict
        output_dict = {}
        outputs = [np.rollaxis(frame_mask_1[0], 3, 0).astype(np.float32),  # [C, H, W, D]
                np.rollaxis(frame_mask_1[1], 3, 0).astype(np.float32),
                xfm_1[:3,:3].astype(np.float32)]
        for name, output in zip(self.output_names, outputs):
            output_dict[name] = torch.tensor(output)  # now tensor (still of shape [C, H, W, D])

        self.iter_idx += 1
        return output_dict

    def __iter__(self):
        self.iter_idx = 0  # reset at the start of every epoch
        return self

    def __len__(self):
        return self.n_samples

    def next(self):
        return self.__next__()

class SpatialAugmenter(object):

    def __init__(self,
                 rotation_range=0.,
                 shift_range=0.,
                 seed=None,
                 return_affine=False,
                 normalise=False,
                 centering=True):
        """Rotation range in degrees, shift range in voxels."""

        # initialisation
        self.rotation_range = rotation_range
        self.shift_range = shift_range
        self.return_affine = return_affine
        self.normalise = normalise
        self.centering = centering

        # enforce numpy seed
        self.rng = np.random.RandomState(seed)

    def random_transform(self, *args):
        """Randomly rotate/translate an image tensor of shape [H, W, D, C], and its labels."""

        # get transformation matrix
        transform_matrix = self.create_random_transform()

        return self.perform_transform(transform_matrix, *args)
    
    def create_random_transform(self):
        """Randomly rotate/translate an image tensor of shape [H, W, D, C], and its labels."""

        # get transformation matrix
        if self.rotation_range:
            rx, ry, rz = self.rng.uniform(-self.rotation_range, self.rotation_range, 3)
        else:
            rx, ry, rz = 0, 0, 0
        if self.shift_range:
            tx, ty, tz = self.rng.uniform(-self.shift_range, self.shift_range, 3)
        else:
            tx, ty, tz = 0, 0, 0
        transform_matrix = utils.create_transform(rx, ry, rz, tx, ty, tz, ordering='txyz')

        return transform_matrix

    def perform_transform(self, transform_matrix, *args):
        # apply transform, computation is done with torch but inputs and outputs are numpy
        with torch.no_grad():
            outputs = []
            for vol_idx, x in enumerate(args):
                if vol_idx > 0:
                    method = "nearest"
                    dtype = torch.int32
                else:
                    method = "linear"
                    dtype = torch.float32
                grid = utils.aff_to_field(transform_matrix, x.shape[:3], invert_affine=True, centering=self.centering)
                x = utils.interpolate(x, grid, method=method, vol_dtype=dtype)
                outputs.append(x)

            if self.normalise:
                outputs[0] = torch.clamp(outputs[0], 0)
                m = torch.min(outputs[0])
                M = torch.max(outputs[0])
                outputs[0] = (outputs[0] - m) / (M - m + 1e-9)

            outputs = [out.detach().numpy() for out in outputs]

        # return outputs
        outputs = [outputs]
        if self.return_affine:
            outputs.append(transform_matrix)
        return outputs[0] if len(outputs) == 1 else outputs  # [H, W, D, C]

class IntensityAugmenter(object):

    def __init__(self,
                 augm_params,
                 seed=None):

        # initialise
        self.aug_model = augmentation.RotE3CNNAugmentation(
            max_bias=augm_params["max_bias"],
            img_res=augm_params["img_res"],
            max_res_iso=augm_params["max_res_iso"],
            gamma=augm_params["gamma"],
            norm_perc=augm_params["norm_perc"]
        )
        self.trainT = self.aug_model.get_transform()
        self.evalT = self.aug_model.get_eval_transform()

        # enforce numpy seed
        self.rng = np.random.RandomState(seed)

    def training_transform(self, imgs, label):
        """Randomly corrupt an image tensor of shape [H, W, D, C] with noise/bias """
        img = torch.tensor(imgs).float().permute(3, 0, 1, 2)
        label = torch.tensor(label).int().permute(3, 0, 1, 2)
        
        subj = torchio.Subject(
            image=torchio.ScalarImage(tensor=img),
            label=torchio.LabelMap(tensor=label)
        )
        subj = self.trainT(subj)
        out = subj['image'].tensor.permute(1, 2, 3, 0).detach().cpu().numpy()
        out_label = subj['label'].tensor.permute(1, 2, 3, 0).detach().cpu().numpy()

        return out, out_label
    
    def eval_transform(self, imgs, label):
        img = torch.tensor(imgs).float().permute(3, 0, 1, 2)
        label = torch.tensor(label).int().permute(3, 0, 1, 2)
        
        subj = torchio.Subject(
            image=torchio.ScalarImage(tensor=img),
            label=torchio.LabelMap(tensor=label)
        )
        subj = self.evalT(subj)
        out = subj['image'].tensor.permute(1, 2, 3, 0).detach().cpu().numpy()
        out_label = subj['label'].tensor.permute(1, 2, 3, 0).detach().cpu().numpy()
        return out, out_label