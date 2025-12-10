# python imports
import pytorch_lightning as pl
import glob

# project imports
from . import unet, dataset, lightning

def training(train_image_dir,
             train_labels_dir,
             val_image_dir,
             val_labels_dir,
             model_dir,
             batchsize=1,
             crop_size=128,
             flipping=False,
             scaling_lower_bound=.5,
             scaling_upper_bound=1.3,
             rotation_bounds=180,
             translation_bounds=10,
             randomise_res=True,
             img_res=3.,
             max_res_iso=8.,
             max_bias=.5,
             noise_std=0.03,
             norm_perc=0.005,
             gamma_min=-0.8,
             gamma_max=0.,
             sigma_min=1.5,
             sigma_max=2.3,
             alpha_min=0.5,
             alpha_max=1.5,
             n_levels=4,
             unet_feat_count=16,
             feat_multiplier=2,
             activation='elu',
             n_output_channels=3,
             lr=1e-4,
             weight_decay=0,
             class_weights=[],
             dice_weight=1.,
             resume=False,
             n_epochs=500):

    # create datasets
    augm_params = {
        'crop_size': crop_size,
        'img_res': img_res,
        'flipping': flipping,
        'scaling_bounds': (scaling_lower_bound, scaling_upper_bound),
        'rotation_bounds': rotation_bounds,
        'translation_bounds': translation_bounds,
        'randomise_res': randomise_res,
        'max_res_iso': max_res_iso,
        'max_bias': max_bias,
        'noise_std': noise_std,
        'norm_perc': norm_perc,
        'gamma': (gamma_min,gamma_max),
        'sigma': (sigma_min, sigma_max),
        'alpha': (alpha_min, alpha_max)
    }
    train_dataset = dataset.SegmentationDataset(
        image_dir=train_image_dir,
        label_dir=train_labels_dir,
        augm_params=augm_params
    )
    val_dataset = dataset.SegmentationDataset(
        image_dir=val_image_dir,
        label_dir=val_labels_dir,
        augm_params=augm_params,
        eval_mode=True
    )

    datasets = {
        'train': train_dataset,
        'val': val_dataset
    }
    
    # create network and lightning module
    net = unet.UNet(
        n_input_channels=1,
        n_output_channels=n_output_channels,
        n_levels=n_levels,
        n_conv=2,
        n_feat=unet_feat_count,
        feat_mult=feat_multiplier,
        kernel_size=3,
        activation=activation,
        last_activation=None
    )
    module_config = {
        'batch_size': batchsize,
        'lr': lr,
        'weight_decay': weight_decay,
        'class_weights': class_weights,
        'dice_weight': dice_weight,
    }
    module = lightning.SegmentationLightningModule(
        config=module_config,
        datasets=datasets,
        model=net,
        log_dir=model_dir
    )
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        monitor="val_loss",
        mode="min",
        dirpath=model_dir,
        filename="model-{epoch:02d}-{val_loss:.2f}",
    )
    # build trainer
    if resume:
        resume_ckpt = glob.glob(f"{model_dir}/model*.ckpt")[0]
    else:
        resume_ckpt = None
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=n_epochs,
        default_root_dir=model_dir,
        callbacks=[checkpoint_callback]
    )

    # train model
    if resume:
        module.load_metrics()
    trainer.fit(module, ckpt_path=resume_ckpt)
    module.save_metrics()
