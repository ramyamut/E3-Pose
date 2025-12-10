import torch
import torchio

from .. import utils, augmentation

class SegmentationDataset(torch.utils.data.Dataset):
    """takes in a directory for images and corresponding masks and creates a dataset for training a segmentation network
    """

    def __init__(self,
                 image_dir,
                 label_dir,
                 augm_params=None,
                 eval_mode=False):

        # input data
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.path_images, self.path_labels = utils.get_image_label_paths(self.image_dir, self.label_dir)

        self.eval_mode = eval_mode
        self.augmenter = augmentation.SegUNetAugmentation(**augm_params)
        
        # define augmentation model
        if self.eval_mode:
            self.transform = self.augmenter.get_val_transform()
        else:
            self.transform = self.augmenter.get_transform()
        
        # define subject dataset
        self.subjects_list = [
            torchio.Subject(
                image=torchio.ScalarImage(img_path), label=torchio.LabelMap(label_path), image_path=img_path, label_path=label_path
            ) for img_path, label_path in zip(self.path_images, self.path_labels)
        ]
        self.n_samples = len(self.subjects_list)
        self.subjects_dataset = torchio.data.SubjectsDataset(self.subjects_list, transform=self.transform)

    def __iter__(self):
        self.iter_idx = 0  # reset at the start of every epoch
        return self

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):

        subject = self.subjects_dataset[idx]
        label = subject['label'].tensor.to(torch.int32)
        
        output_dict = {
            'image': subject['image'].tensor.to(torch.float32),
            'label': label,
            'image_path': subject['image_path'],
            'label_path': subject['label_path']
        }

        return output_dict
