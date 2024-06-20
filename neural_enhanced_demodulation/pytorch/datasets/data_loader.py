# data_loader.py

import os
import torch
from torch.utils.data import DataLoader
from torch.utils import data
import torch.nn.functional as F
from scipy.ndimage.filters import uniform_filter1d 
import math


import scipy.io as scio
import numpy as np
from PIL import Image

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

        return len(self.data_lists)

    def __getitem__(self, index):
        'Generates one sample of data'
        data_file_name = self.data_lists[index]
        with open(data_file_name, 'rb') as fid:
            nsamp = self.opts.n_classes * self.opts.fs // self.opts.bw 
            lora_img = torch.tensor(np.fromfile(fid, np.complex64, nsamp), dtype = torch.cfloat)
            assert lora_img.shape[0] == nsamp, data_file_name
 
        if not self.groundtruth:
                nsamp = self.opts.n_classes * self.opts.fs // self.opts.bw 
                mwin = nsamp//2
                datain = lora_img[:]
                A = uniform_filter1d(abs(datain),size=mwin) 
                datain = datain[A >= max(A)/2] 
                amp_sig = torch.mean(torch.abs(torch.tensor(datain))) 
                 
                amp = math.pow(0.1, self.opts.snr/20) * amp_sig
                noise =  torch.tensor(amp / math.sqrt(2) * np.random.randn(nsamp) + 1j * amp / math.sqrt(2) * np.random.randn(nsamp), dtype = torch.cfloat)

                lora_img = lora_img + noise

        data_per = torch.tensor(lora_img, dtype=torch.cfloat)

        label_per = data_file_name[:-4]
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
