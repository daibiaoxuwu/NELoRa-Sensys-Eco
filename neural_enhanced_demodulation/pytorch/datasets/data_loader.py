# data_loader.py

import os
import random
import math

import torch
from torch.utils.data import DataLoader
from torch.utils import data
from scipy.ndimage.filters import uniform_filter1d 
import numpy as np

class lora_dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, opts, files_list, groundtruth=False):
        'Initialization'
        # self.scaling_for_intensity = opts.scaling_for_intensity
        self.opts = opts
        self.data_dir = opts.data_dir
        self.data_lists = files_list
        self.groundtruth = groundtruth
        self.groundtruth_code = opts.groundtruth_code

    def __len__(self):
        'Denotes the total number of samples'

        return len(self.data_lists) * len(self.opts.snr_list)

    def __getitem__(self, index):
        'Generates one sample of data'
        data_file_name = self.data_lists[index % len(self.data_lists)]
        with open(data_file_name, 'rb') as fid:
            nsamp = self.opts.n_classes * self.opts.fs // self.opts.bw 
            lora_img = torch.tensor(np.fromfile(fid, np.complex64, nsamp), dtype = torch.cfloat)
            assert lora_img.shape[0] == nsamp, data_file_name
 
        if not self.groundtruth:
            snr = self.opts.snr_list[- index // len(self.data_lists)] # to make SNR from high to low in each epoch
            nsamp = self.opts.n_classes * self.opts.fs // self.opts.bw 
            mwin = nsamp//2
            datain = lora_img[:]
            A = uniform_filter1d(abs(datain),size=mwin) 
            datain = datain[A >= max(A)/2] 
            amp_sig = torch.mean(torch.abs(torch.tensor(datain))) 
             
            amp = math.pow(0.1, snr/20) * amp_sig
            noise =  torch.tensor(amp / math.sqrt(2) * np.random.randn(nsamp) + 1j * amp / math.sqrt(2) * np.random.randn(nsamp), dtype = torch.cfloat)

            lora_img = lora_img + noise
        else:
            snr = self.opts.groundtruth_code

        data_per = torch.tensor(lora_img, dtype=torch.cfloat)

        label_per = os.path.splitext(data_file_name)[0] + '_' + str(snr)
        return data_per, label_per


# receive the csi feature map derived by the ray model as the input
def lora_loader(opts, files_train, files_test, groundtruth):
    """Creates training and test data loaders.
    """

    training_dataset = lora_dataset(opts, files_train, groundtruth)
    testing_dataset = lora_dataset(opts, files_test, groundtruth)

    training_dloader = DataLoader(dataset=training_dataset,
                                  batch_size=opts.batch_size,
                                  shuffle=False,
                                  num_workers=opts.num_workers)
    testing_dloader = DataLoader(dataset=testing_dataset,
                                 batch_size=opts.batch_size,
                                 shuffle=False,
                                 num_workers=opts.num_workers)
    return training_dloader, testing_dloader

