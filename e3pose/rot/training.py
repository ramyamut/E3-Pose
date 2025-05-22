import os
import re
import pandas as pd
import numpy as np
import torch.utils.data
from shutil import copy2
from pytorch3d.transforms import matrix_to_euler_angles
import nibabel as nib

from e3pose.rot import networks, loaders
from e3pose import utils

# set up cuda and device
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def training(training_im_dir,
             training_lab_dir,
             training_anno_csv,
             val_im_dir,
             val_lab_dir,
             val_anno_csv,
             results_dir,
             image_size=64,
             rotation_range=90,
             shift_range=5,
             norm_perc=0.005,
             n_levels=4,
             kernel_size=5,
             batch_size=1,
             learning_rate=0.01,
             weight_decay=3e-5,
             momentum=0.99,
             n_epochs=100000,
             validate_every_n_epoch=100,
             resume=False):

    # reformat inputs
    image_size = [image_size] * 3 if not isinstance(image_size, list) else image_size

    # create result directory
    models_dir = os.path.join(results_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # set up augmenter params
    augment_params = {'resize': image_size,
                      'rotation_range': rotation_range,
                      'shift_range': shift_range,
                      'norm_perc': norm_perc}

    # training loader
    print('create training loader...', flush=True)
    training_subj_dict = utils.build_subject_frame_dict_all(training_im_dir, training_lab_dir)
    training_anno_df = pd.read_csv(training_anno_csv)
    train_dataset = loaders.loader_rot_canonical(subj_dict=training_subj_dict,
                                        anno_df=training_anno_df,
                                        augm_params=augment_params,
                                        seed=1919)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    # validation loader
    print('create validation loader...', flush=True)
    val_subj_dict = utils.build_subject_frame_dict_all(val_im_dir, val_lab_dir)
    val_anno_df = pd.read_csv(val_anno_csv)
    val_dataset = loaders.loader_rot_canonical(subj_dict=val_subj_dict,
                                      anno_df=val_anno_df,
                                      augm_params=augment_params,
                                      eval=True,
                                      seed=1919)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

    # initialise feature extractor
    print('initialise architecture...', flush=True)
    net = networks.E3CNN_Encoder(input_chans=1, output_chans=1, n_levels=n_levels, k=kernel_size, last_activation=None, equivariance='O3')
    net = net.to(device)

    # initialise optimizer
    print('initialise optimizer...\n', flush=True)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

    # Check if need to resume training and load weights
    last_epoch = 0
    best_val_loss = 1e9
    best_val_dice = 0
    list_scores = []
    if resume:
        previous_files = sorted([p for p in os.listdir(models_dir) if re.sub('\D', '', p) != ''])
        if len(previous_files) > 0:
            print(f'loading from {previous_files[-1]}', flush=True)
            checkpoint = torch.load(os.path.join(models_dir, previous_files[-1]), map_location=torch.device(device))
            net.load_state_dict(checkpoint['net_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            last_epoch = checkpoint['epoch']
            path_val_scores = os.path.join(results_dir, 'val_scores.npy')
            if os.path.isfile(path_val_scores):
                list_scores = np.load(path_val_scores)
                best_val_loss = np.min(list_scores[:, 2])
                best_val_dice = np.max(list_scores[:, 3])
                list_scores = list_scores.tolist()

    # Training loop
    sample = 0
    for epoch in range(last_epoch, n_epochs):
        print('Epoch', epoch, flush=True)

        net.train()
        epoch_train_loss = 0
        epoch_train_err_R = 0
        for i, batch in enumerate(train_loader):

            # obtain sample from loader
            image = batch['image'].to(device)
            mask = batch['mask'].to(device)
            rot = batch['rot'].to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):

                # get predictions
                output = net.forward(image, pool=False)
                pred_norm = output.reshape(image.shape[0], -1, 3, output.shape[2]*output.shape[3]*output.shape[4])
                pred_norm = torch.nn.functional.normalize(pred_norm,dim=2) #[B, 3, 3, vox]
                target = torch.linalg.inv(rot)[:,:3,:3].permute(0,2,1).unsqueeze(-1) #[B, 3, 3, 1]

                # compute loss
                pseudovector_loss = torch.cross(pred_norm[:,0], target[:,0], dim=1).norm(dim=1).mean()
                vector_loss = torch.nn.functional.l1_loss(pred_norm[:,1:], target[:,1:])
                train_loss = 0.5*pseudovector_loss + vector_loss
                
                with torch.no_grad():
                # compute rotation error
                    pred = torch.nn.functional.normalize(net.pool(output)[:,:,0,0,0].reshape(image.shape[0], -1, 3))
                    rot_pred = utils.axes_to_rotation(pred[:,0], pred[:,1], pred[:,2])
                    err_R = utils.rotation_matrix_to_angle_loss(rot, rot_pred).item()
                
                epoch_train_err_R += err_R
                    
                # backpropagation
                train_loss.backward(retain_graph=True)
                sample += 1
                optimizer.step()

            # print iteration info
            epoch_train_loss += train_loss.item() * batch_size
            print('iteration:{}/{}  loss:{:.5f}'.format(i + 1, len(train_dataset), train_loss.item()), flush=True)

            # flush cuda memory
            del image, mask, rot, train_loss
            torch.cuda.empty_cache()

        epoch_train_loss = epoch_train_loss / len(train_dataset)
        epoch_train_err_R = epoch_train_err_R / len(train_dataset)

        # save and validate model every validate_every_n_epoch epochs, otherwise just print training info
        if epoch % validate_every_n_epoch != 0:
            print('Epoch:{}  Train Loss:{:.5f}'.format(epoch, epoch_train_loss) + '\n', flush=True)

        else:
            torch.save({'net_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch},
                       os.path.join(models_dir, '%05d.pth' % epoch))

            # eval loop
            net.eval()
            epoch_val_loss = 0
            epoch_val_err_R = 0

            for j, batch in enumerate(val_loader):

                image = batch['image'].to(device)
                mask = batch['mask'].to(device)
                rot = batch['rot'].to(device)
                
                output = net.forward(image, pool=False)
                pred_norm = output.reshape(image.shape[0], -1, 3, output.shape[2]*output.shape[3]*output.shape[4])
                pred_norm = torch.nn.functional.normalize(pred_norm,dim=2) #[B, 3, 3, vox]
                target = torch.linalg.inv(rot)[:,:3,:3].permute(0,2,1).unsqueeze(-1) #[B, 3, 3, 1]

                # compute loss
                pseudovector_loss = torch.cross(pred_norm[:,0], target[:,0], dim=1).norm(dim=1).mean()
                vector_loss = torch.nn.functional.l1_loss(pred_norm[:,1:], target[:,1:])
                val_loss = 0.5*pseudovector_loss + vector_loss
                
                epoch_val_loss += val_loss.item()
                
                # compute rotation error
                pred = torch.nn.functional.normalize(net.pool(output)[:,:,0,0,0].reshape(image.shape[0], -1, 3))
                rot_pred = utils.axes_to_rotation(pred[:,0], pred[:,1], pred[:,2])
                err_R = utils.rotation_matrix_to_angle_loss(rot, rot_pred).item()
                epoch_val_err_R += err_R

                # flush cuda memory
                del image, mask, rot, val_loss
                torch.cuda.empty_cache()

            # save validation scores
            epoch_val_loss = epoch_val_loss / len(val_dataset)
            epoch_val_err_R = epoch_val_err_R / len(val_dataset)
            print('Epoch:{}  Train Loss:{:.5f}  err R:{:.3f} '.format(
                epoch, epoch_train_loss, epoch_train_err_R) + '\n', flush=True)
            print('Epoch:{}  Val Loss:{:.5f}  err R:{:.3f} '.format(
                epoch, epoch_val_loss, epoch_val_err_R) + '\n', flush=True)
            list_scores.append([epoch, epoch_train_loss, epoch_train_err_R, epoch_val_loss, epoch_val_err_R])
            np.save(os.path.join(results_dir, 'val_scores.npy'), np.array(list_scores))

            # save best model
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                copy2(os.path.join(models_dir, '%05d.pth' % epoch), os.path.join(results_dir, 'best_epoch_val_loss.pth'))
                with open(os.path.join(results_dir, 'best_epoch_val_loss.txt'), 'w') as f:
                    f.write('epoch:%d   val loss:%f' % (epoch, best_val_loss))

    del net

