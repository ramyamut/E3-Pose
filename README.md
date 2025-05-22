# E3-Pose

In this repository, we present E3-Pose, the first framework for 3D canonical rigid pose estimation from volumetric medical images that uses an $E(3)$-equivariant convolutional neural network (E3-CNN). Although we evaluate the utility of E3-Pose on fetal brain MRI for the application of automated slice prescription, the proposed methods are applicable more broadly.

We rapidly estimate pose from volumes in a two-step process that separately estimates translation and rotation:

1. Translation Estimation:
    * A standard segmentation U-Net localizes the object in the volume.
    * The center-of-mass (CoM) of the predicted mask is the estimated origin of the canonical coordinate frame. 
2. Rotation Estimation:
    * We crop input volumes such that the predicted segmentation mask is scaled to 60% of the cropped dimensions.
    * The E3-CNN takes in the cropped volume as input, and outputs a 9D rotation representation consisting of 2 vectors and 1 pseudovector.
    * The final output rotation to the canonical coordinate frame is computed by choosing the pseudovector direction that ensures right-handedness, and orthonormalizing via support-vector decomposition (SVD).

Our E3-CNN architecture builds on prior theoretical work on [3D steerable CNNs](https://proceedings.neurips.cc/paper_files/paper/2018/file/488e4104520c6aab692863cc1dba45af-Paper.pdf) and uses code borrowed from [e3nn-UNet](https://github.com/SCAN-NRAD/e3nn_Unet), which implements 3D convolutions with the [e3nn](https://e3nn.org/) Python library for building $E(3)$-equivariant networks.

<br />

![Method overview](images/method_overview.png)

<br />

Overall, E3-Pose outperforms state-of-the-art methods for rigid pose estimation in fetal brain MRI, including strategies that rely on classical optimisation ([ANTs](https://github.com/ANTsX/ANTs)<sup>1</sup>), anatomical landmark detection ([Fetal-Align](https://github.com/mu40/fetal-align)), keypoint detection ([EquiTrack](https://github.com/BBillot/EquiTrack)), and direct pose regression with standard CNNs ([6DRep](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12464/124640T/Automatic-brain-pose-estimation-in-fetal-MRI/10.1117/12.2647613.full), [RbR](https://github.com/HuXiaoling/Regre4Regis)). See figure below for rigid pose estimation in low-SNR volumes with severe artifacts. Particularly, we show in our paper that regularizing network parameters to conform with the physical symmetry of rigid pose estimation mitigates overfitting and permits better generalization to out-of-distribution data.

<br />

![vNav examples](images/vnav_examples.png)

<br />

The full article describing this method is available at:

**E3-Pose: Equivariant Rigid Pose Estimation With Application to Slice Prescription in Fetal Brain MRI** \
Muthukrishnan, Billot, Gagoski, Soldatelli, Grant, Golland


---
### Installation

1. Clone this repository.
2. Install python 3.10.
3. Install all the [required libraries](requirements.txt).
4. If you want to use our trained model weights for fetal brain MRI, download the model weights [here](https://drive.google.com/drive/folders/1r6FVzXG9VLH-0MtMnD2hwjzdDqss1DSE?usp=sharing).

You're now ready to use E3-Pose!

<br />

---
### Usage

This repository contains all the code necessary to train and test your own networks. We provide separate scripts for training the segmentation U-Net and E3-CNNs, and a single script to deploy both for full rigid pose estimation.

#### Training a Segmentation U-Net for Translation Estimation

1. Set up separate training/validation dataset directories for images and ground-truth segmentation labels, where file names between image and label directories are the same. Ensure that all image file extensions are .nii or .nii.gz.

2. If you are training a multi-class segmentation network, ensure that the object for which you want to estimate pose has category label 1 in the ground-truth labels.

3. Name the output directory to save all model weights and metrics during network training.

4. To train the segmentation U-Net, run:

    ```
    python scripts/train_unet.py train_image_dir/ train_label_dir/ val_image_dir/ val_label_dir/ output_dir/
    ```

    For detailed descriptions of other arguments, run:
    
    ```
    python scripts/train_unet.py -h
    ```

#### Training an E3-CNN for Rotation Estimation

1. Set up separate training/validation dataset directories for images and ground-truth segmentation labels, where file names between image and label directories are the same. Ensure that all image file extensions are .nii or .nii.gz. If your segmentation labels have multiple classes, ensure that the object for which you want to estimate pose has category label 1.

2. Set up separate CSV files for rotation annotations in training and validation datasets, in the following format:

    | frame_id | rot_x | rot_y | rot_z |
    |----------|-------|-------|-------|
    | ...      | ...   | ...   | ...   |

    where **frame_id** is the file name of the volume without the file extension, and **rot_x**, **rot_y**, **rot_z** are the Euler angles in degrees of the rotation from the volume to the canonical coordinate frame. The Euler angle rotation assumes the "xyz" ordering convention.

3. Name the output directory to save all model weights and metrics during network training.

4. To train the E3-CNN, run:

    ```
    python scripts/train_e3cnn.py train_image_dir/ train_label_dir/ path_to_train_annotations.csv \
        val_image_dir/ val_label_dir/ path_to_val_annotations.csv \
        output_dir/
    ```

    For detailed descriptions of other arguments, run:
    
    ```
    python scripts/train_e3cnn.py -h
    ```

- [inference.py](scripts/inference.py) explains how to run inference on an input volume given trained model weights for both networks.

<br />

---
### Citation/Contact

This code is under [Apache 2.0](LICENSE.txt) licensing.

If you find this work useful for your research, please cite:

**E3-Pose: Equivariant Rigid Pose Estimation With Application to Slice Prescription in Fetal Brain MRI** \
Muthukrishnan, Billot, Gagoski, Soldatelli, Grant, Golland

If you have any question regarding the usage of this code, or any suggestions to improve it, please raise an issue
(preferred) or contact us at:\
**ramyamut@mit.edu**


<br />

---
### References

<sup>1</sup> *3D Steerable CNNs: Learning Rotationally Equivariant Features in Volumetric Data* \
Weiler, Geiger, Welling, Boomsma, Cohen \
Advances in Neural Information Processing Systems, 2018

<sup>2</sup> *Leveraging SO(3)-steerable convolutions for pose-robust semantic segmentation in 3D medical data* \
Diaz, Geiger, McKinley \
Journal of Machine Learning in Biomedical Imaging, 2024

<sup>3</sup> *e3nn: Euclidean neural networks* \
Geiger and Smidt \
arXiV, 2022

<sup>4</sup> *A reproducible evaluation of ANTs similarity metric performance in brain image registration* \
Avants, Tustison, Song, Cook, Klein, Gee \
NeuroImage, 2011

<sup>5</sup> *Rapid head-pose detection for automated slice prescription of fetal-brain MRI* \
Hoffmann, Abaci Turk, Gagoski, Morgan, Wighton, Tisdall, Reuter, Adalsteinsson, Grant, Wald, van der Kouwe \
International Journal of Imaging Systems and Technology, 2021

<sup>6</sup> *SE(3)-Equivariant and Noise-Invariant 3D Rigid Motion racking in Brain MRI* \
Billot, Dey, Moyer, Hoffmann, Abaci Turk, Gagoski \
IEEE Transactions on Medical Imaging, 2024

<sup>7</sup> *Automatic brain pose estimation in fetal MRI* \
Faghihpirayesh, Karimi, Erdogmus, Gholipour \
Proceedings of SPIE: Medical Imaging: Image Processing, 2023

<sup>8</sup> *Registration by Regression (RbR): a framework for interpretable and flexible atlas registration* \
Gopinathm Hum Hoffmnn, Puonti, Iglesias \
International Workshop on Biomedical Image Registration, 2024
