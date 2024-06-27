# end2end.py

from __future__ import division
import os

import warnings

warnings.filterwarnings("ignore")

# Torch imports
import torch
import torch.fft
import torch.nn as nn
import torch.optim as optim

# Numpy & Scipy imports
import numpy as np
import scipy.io
import torchvision.utils as vutils
from PIL import Image
import math

import cv2
# Local imports
from utils import to_var, to_data, spec_to_network_input
from models.model_components import maskCNNModel, classificationHybridModel
import torch.autograd.profiler as profiler
import time

SEED = 11

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def print_models(Model):
    """Prints model information for the generators and discriminators.
    """
    print("                 Model                ")
    print("---------------------------------------")
    print(Model)
    print("---------------------------------------")


def create_model(opts):
    """Builds the generators and discriminators.
    """

    maskCNN = maskCNNModel(opts)

    C_XtoY = classificationHybridModel(conv_dim_in=opts.y_image_channel,
                                       conv_dim_out=opts.n_classes,
                                       conv_dim_lstm=opts.conv_dim_lstm)

    if torch.cuda.is_available():
        maskCNN.cuda()
        C_XtoY.cuda()
        print('Models moved to GPU.')

    return maskCNN, C_XtoY


def checkpoint(iteration, mask_CNN, C_XtoY, opts):
    """Saves the parameters of both generators G_YtoX, G_XtoY and discriminators D_X, D_Y.
    """

    mask_CNN_path = os.path.join(opts.checkpoint_dir, str(iteration) + '_maskCNN.pkl')
    torch.save(mask_CNN.state_dict(), mask_CNN_path)

    C_XtoY_path = os.path.join(opts.checkpoint_dir, str(iteration) + '_C_XtoY.pkl')
    torch.save(C_XtoY.state_dict(), C_XtoY_path)


def load_checkpoint(opts):
    """Loads the generator and discriminator models from checkpoints.
    """

    maskCNN_path = os.path.join(opts.checkpoint_dir, str(opts.load_iters) + '_maskCNN.pkl')

    maskCNN = maskCNNModel(opts)

    maskCNN.load_state_dict(torch.load(
        maskCNN_path, map_location=lambda storage, loc: storage),
        strict=False)

    C_XtoY_path = os.path.join(opts.checkpoint_dir, str(opts.load_iters) + '_C_XtoY.pkl')

    C_XtoY = classificationHybridModel(conv_dim_in=opts.x_image_channel,
                                       conv_dim_out=opts.n_classes,
                                       conv_dim_lstm=opts.conv_dim_lstm)

    C_XtoY.load_state_dict(torch.load(
        C_XtoY_path, map_location=lambda storage, loc: storage),
        strict=False)

    if torch.cuda.is_available():
        maskCNN.cuda()
        C_XtoY.cuda()
        print('Models moved to GPU.')

    return maskCNN, C_XtoY

def save_samples(iteration, sources, targets, references, opts, nameX, name):

    # Convert complex images to magnitude
    def convert_to_magnitude(image_tensor):
        return torch.angle(image_tensor)
        real = image_tensor[:, 0, :, :]
        imag = image_tensor[:, 1, :, :]
        magnitude = torch.sqrt(real ** 2 + imag ** 2)
        return magnitude

    # Convert the batch of tensors to magnitudes
    sources_magnitude = convert_to_magnitude(sources)
    targets_magnitude = convert_to_magnitude(targets)
    references_magnitude = convert_to_magnitude(references)

    # Concatenate images along the width
    concatenated_images = []
    for i in range(opts.batch_size):
        concatenated = torch.cat((sources_magnitude[i], targets_magnitude[i], references_magnitude[i]), dim=1)
        concatenated_images.append(concatenated)

    # Stack all images into a single tensor
    grid = torch.stack(concatenated_images).unsqueeze(1)

    # Convert the grid to a single image and save
    grid_image = vutils.make_grid(grid, nrow=math.ceil(math.sqrt(opts.batch_size)), padding=2, normalize=True)
    ndarr = grid_image.mul(255).byte().permute(1, 2, 0).cpu().numpy()
    im = Image.fromarray(ndarr)

    # Save the image
    snr = int(os.path.basename(nameX).split('_')[-1])
    save_path = os.path.join(opts.sample_dir,f'{name}-{iteration:06d}-snr{snr}-Y.png')
    im.save(save_path)

def training_loop(training_dataloader_X, training_dataloader_Y, testing_dataloader_X,
                  testing_dataloader_Y, opts):
    """Runs the training loop.
        * Saves checkpoint every opts.checkpoint_every iterations
        * Saves generated samples every opts.sample_every iterations
    """
    loss_spec = torch.nn.MSELoss(reduction='mean')
    loss_class = nn.CrossEntropyLoss()
    # Create generators and discriminators
    if opts.load:
        mask_CNN, C_XtoY = load_checkpoint(opts)
    else:
        mask_CNN, C_XtoY = create_model(opts)

    g_params = list(mask_CNN.parameters()) + list(C_XtoY.parameters())
    g_optimizer = optim.Adam(g_params, opts.lr, [opts.beta1, opts.beta2])

    iter_X = iter(training_dataloader_X)
    iter_Y = iter(training_dataloader_Y)

    test_iter_X = iter(testing_dataloader_X)
    test_iter_Y = iter(testing_dataloader_Y)

    # Get some fixed data from domains X and Y for sampling. These are images that are held
    # constant throughout training, that allow us to inspect the model's performance.
    fixed_X, name_X_fixed = next(test_iter_X)
    fixed_X = to_var(fixed_X)

    fixed_Y, name_Y_fixed = next(test_iter_Y)
    fixed_Y = to_var(fixed_Y)
    # print("Fixed_X {}".format(fixed_X.shape))
    fixed_X_spectrum_raw = torch.stft(input=fixed_X, n_fft=opts.stft_nfft, hop_length=opts.stft_overlap,
                                      win_length=opts.stft_window, pad_mode='constant');
    fixed_X_spectrum = spec_to_network_input(fixed_X_spectrum_raw, opts)
    # print("Fixed {}".format(fixed_X_spectrum.shape))

    fixed_Y_spectrum_raw = torch.stft(input=fixed_Y, n_fft=opts.stft_nfft, hop_length=opts.stft_overlap,
                                      win_length=opts.stft_window, pad_mode='constant');
    fixed_Y_spectrum = spec_to_network_input(fixed_Y_spectrum_raw, opts)

    iter_per_epoch = min(len(iter_X), len(iter_Y))

    for iteration in range(1, opts.train_iters + 1):
        if iteration % iter_per_epoch == 0:
            iter_X = iter(training_dataloader_X)
            iter_Y = iter(training_dataloader_Y)

        images_X, name_X = next(iter_X)
        labels_X_mapping = list(
            map(lambda x: int(os.path.basename(x).split('_')[1]), name_X))
        images_X, labels_X = to_var(images_X), to_var(
            torch.tensor(labels_X_mapping))
        images_Y, name_Y = next(iter_Y)
        labels_Y_mapping = list(
            map(lambda x: int(os.path.basename(x).split('_')[1]), name_Y))
        images_Y, labels_Y = to_var(images_Y), to_var(
            torch.tensor(labels_Y_mapping))

        # ============================================
        #            TRAIN THE GENERATOR
        # ============================================

        images_X_spectrum_raw = torch.stft(input=images_X, n_fft=opts.stft_nfft, hop_length=opts.stft_overlap,

                                           win_length=opts.stft_window, pad_mode='constant');
        images_X_spectrum = spec_to_network_input(images_X_spectrum_raw, opts)

        images_Y_spectrum_raw = torch.stft(input=images_Y, n_fft=opts.stft_nfft, hop_length=opts.stft_overlap,
                                           win_length=opts.stft_window, pad_mode='constant');
        images_Y_spectrum = spec_to_network_input(images_Y_spectrum_raw, opts)
        #########################################
        ##    FILL THIS IN: X--Y               ##
        #########################################
        if iteration % 50 == 0:
            print("Iteration: {}/{}".format(iteration, opts.train_iters))
        fake_Y_spectrum = mask_CNN(images_X_spectrum)
        # 2. Compute the generator loss based on domain Y
        g_y_pix_loss = loss_spec(fake_Y_spectrum, images_Y_spectrum)
        labels_X_estimated = C_XtoY(fake_Y_spectrum)
        g_y_class_loss = loss_class(labels_X_estimated, labels_X)
        g_optimizer.zero_grad()
        G_Image_loss = opts.scaling_for_imaging_loss * g_y_pix_loss
        G_Class_loss = opts.scaling_for_classification_loss * g_y_class_loss
        G_Y_loss = G_Image_loss + G_Class_loss
        G_Y_loss.backward()
        g_optimizer.step()

        # Print the log info
        if iteration % opts.log_step == 0:
            print(
                'Iteration [{:5d}/{:5d}] | G_Y_loss: {:6.4f}| G_Image_loss: {:6.4f}| G_Class_loss: {:6.4f}'
                    .format(iteration, opts.train_iters,
                            G_Y_loss.item(),
                            G_Image_loss.item(),
                            G_Class_loss.item()))

        # Save the generated samples
        if (iteration % opts.sample_every == 0) and (not opts.server):
            save_samples(iteration,fixed_X_spectrum,fake_Y_spectrum,fixed_Y_spectrum, opts, name_X_fixed[0], 'fix')

        # Save the model parameters
        if iteration % opts.checkpoint_every == 0:
            checkpoint(iteration, mask_CNN, C_XtoY, opts)

    test_iter_X = iter(testing_dataloader_X)
    test_iter_Y = iter(testing_dataloader_Y)
    iter_per_epoch_test = min(len(test_iter_X), len(test_iter_Y))

    error_matrix = np.zeros([len(opts.snr_list), 1], dtype=float)
    error_matrix_count = np.zeros([len(opts.snr_list), 1], dtype=int)

    error_matrix_info = []

    # iter_per_epoch_test = 500
    saved_data = {}
    for iteration in range(iter_per_epoch_test):
        images_X_test, name_X_test = next(test_iter_X)

        code_X_test_mapping = list(
            map(lambda x: float(os.path.basename(x).split('_')[0]), name_X_test))

        snr_X_test_mapping = list(
            map(lambda x: int(os.path.basename(x).split('_')[-1]), name_X_test))

        instance_X_test_mapping = list(
            map(lambda x: int(os.path.basename(x).split('_')[2]), name_X_test))

        labels_X_test_mapping = list(
            map(lambda x: int(os.path.basename(x).split('_')[1]), name_X_test))

        images_X_test, labels_X_test = to_var(images_X_test), to_var(
            torch.tensor(labels_X_test_mapping))

        images_Y_test, labels_Y_test = next(test_iter_Y)
        images_Y_test = to_var(images_Y_test)

        images_X_test_spectrum_raw = torch.stft(input=images_X_test, n_fft=opts.stft_nfft,
                                                hop_length=opts.stft_overlap, win_length=opts.stft_window,
                                                pad_mode='constant');
        images_X_test_spectrum = spec_to_network_input(images_X_test_spectrum_raw, opts)

        images_Y_test_spectrum_raw = torch.stft(input=images_Y_test, n_fft=opts.stft_nfft,
                                                hop_length=opts.stft_overlap, win_length=opts.stft_window,
                                                pad_mode='constant');
        images_Y_test_spectrum = spec_to_network_input(images_Y_test_spectrum_raw, opts)
        fake_Y_test_spectrum = mask_CNN(images_X_test_spectrum)
        labels_X_estimated = C_XtoY(fake_Y_test_spectrum)
        saved_sample = to_data(labels_X_estimated)

        for i, label in enumerate(to_data(labels_X_test)):
            if label not in saved_data.keys():
                saved_data[label] = []
                saved_data[label].append(saved_sample[i])
            else:
                saved_data[label].append(saved_sample[i])
        _, labels_X_test_estimated = torch.max(labels_X_estimated, 1)

        test_right_case = (labels_X_test_estimated == labels_X_test)
        test_right_case = to_data(test_right_case)

        for batch_index in range(len(test_right_case)):
                snr_index = opts.snr_list.index(snr_X_test_mapping[batch_index])
                error_matrix[snr_index] += test_right_case[batch_index]
                error_matrix_count[snr_index] += 1
                error_matrix_info.append([instance_X_test_mapping[batch_index], code_X_test_mapping[batch_index],
                                          snr_X_test_mapping[batch_index],
                                          labels_X_test_estimated[batch_index].cpu().data.int(),
                                          labels_X_test[batch_index].cpu().data.int()])
        if iteration % opts.log_step == 0: 
            print('Testing Iteration [{:5d}/{:5d}]'
                  .format(iteration, iter_per_epoch_test))
    error_matrix = np.divide(error_matrix, error_matrix_count)
    error_matrix_info = np.array(error_matrix_info)
    print('Accuracy:')
    with open('results.txt', 'a') as f:
        for i in range(len(opts.snr_list)):
            print(opts.snr_list[i], error_matrix[i][0])
            f.write(str(error_matrix) + '\n')
    scipy.io.savemat(
        opts.root_path + '/' + opts.dir_comment + '_' + str(opts.sf) + '_' + str(opts.bw) + '.mat',
        dict(error_matrix=error_matrix,
             error_matrix_count=error_matrix_count,
             error_matrix_info=error_matrix_info))

