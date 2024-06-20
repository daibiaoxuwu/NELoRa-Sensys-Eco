# Neural-enhanced Demodulation

This module provides the codes for our neural-enhanced demodulation, including the preprocessing for symbol transformation and DNN-based demodulation.
Modified. Now generation_dataset.m is not used; dataset generation (processing data from original samples, e.g. raw_sf7_cross_instance) is done on the fly.

# Overview

- [Neural-enhanced Demodulation](#neural-enhanced-demodulation)
- [Overview](#overview)
- [Getting Started](#getting-started)
    - [Dependencies](#dependencies)
    - [Load Dataset](#load-dataset)
- [Run Experiments and Validate Results](#run-experiments-and-validate-results)
    - [From the Scratch](#from-the-scratch)
    - [Direct Inference](#direct-inference)
    - [Evaluation](#evaluation)


# Getting Started

### Dependencies
* scipy==1.0.0
* numpy==1.16.4
* torchvision==0.3.0
* PyTorch==1.7.0

### Load Dataset
1. Clone this repo

2. Download raw_sf7_cross_instance.zip from [here](https://drive.google.com/drive/folders/1iODrhHg6DmSuAGlq5eTSKClybjTLs9ot?usp=sharing)
This is the raw dataset, file size is around 200M. Unzip it.
The other files on the google drive with around 5 Gigabytes are datasets generated from raw_sf7_cross_instance.zip, and since we EMITTED this phase in the improved code in this repo, these large zip files are not used.

Dataset are in 6 folders, which is data collected from 6 different environments.
Each of these 6 folders contain ~30 subfolders. Each subfolder contanins symbols from one packet.
Each file name is:
    {symbol_index} _ {Code} _ {Original_SNR} _ {SF} _ {BW} _ {packet_index}.mat

# Run Experiments and Validate Results

### Direct Inference ###

1. Download the pretrained models from [here](https://drive.google.com/drive/folders/1At3KaE4TojL8YV3YM-DrDpiwmGkiQ--B?usp=sharing)
There are two files 100000_maskCNN.pkl, 100000_C_XtoY.pkl.

2. Specify the root path, data path, and evaluation_dir in `config.py`
<pre>
 ├ [DIR] root_path (.)
     ┬    
     ├ [DIR] evaluations_dir
        |-[DIR] [dir_comment]\_checkpoints
        |-[DIR] [dir_comment]\_samples
        |-[DIR] [dir_comment]\_testing        
 ├ [DIR] data_dir (/path/to/raw_sf7_cross_instance)
</pre>
dir_comment default is 'v0', evaluations_dir default is 'evaluations'.
use the dir_comment to easily manage versions of code and checkpoints and their results.
So default checkpoint_path is './evaluations/v0_checkpoints'. You can see a stub file Put_Your_Downloaded_Checkpoint_Here_By_Default.txt there. 
Put the two files 100000_maskCNN.pkl, 100000_C_XtoY.pkl in the same folder as that txt. (By default dir_comment and evaluations_dir values)

2. run the following command:
```
python main.py --data_dir /path/to/raw_sf7_cross_instance --normalization --train_iter 0 --ratio_bt_train_and_test 0.8 --load --load_iters 100000 
```
Useful parameters:
 --batch_size: batch size,
 --sf: spreading factor,
 --bw: bandwidth,
 --fs: sampling rate,
 --normalization Add this to normalize signals, recommended! Add this flag to get desired results from the saved checkpoint.
 --train_iter how many iterations to train
 --ratio_bt_train_and_test the ratio between training and testing datasets, 0.8 means a 8:2 split
 --load Whether to load a checkpoint from file before training
 --log_step How many steps for each time a log is printed on console
 --sample_every How many steps for each time a sample of the masked spectrograms are saved to \[evaluations_dir\]/\[dir_comment\]\_\[sample_dir\]
 --checkpoint\_every How many steps for each time a checkpoint of the model during training is saved to \[evaluations_dir\]/\[dir_comment\]\_\[checkpoint_dir\]

3. Get a decode result with in your `pytorch/` directory.
The results is also printed to console, in format {SNR} {ACC}
typical result will be:
```
================================================================================
                                      Opts
--------------------------------------------------------------------------------
                            free_gpu_id: -1
                        x_image_channel: 2
                        y_image_channel: 2
                       conv_kernel_size: 3
                      conv_padding_size: 1
                               lstm_dim: 400
                                fc1_dim: 600
                                     sf: 7
                                     bw: 125000
                                     fs: 1000000
                          normalization: 1
                             load_iters: 100000
                             batch_size: 8
                            num_workers: 1
                                     lr: 0.0002
                           sorting_type: -1
               scaling_for_imaging_loss: 128
        scaling_for_classification_loss: 1
                                  beta1: 0.5
                                  beta2: 0.999
                              root_path: ./
                        evaluations_dir: evaluations
                              data_dir: /data/djl/raw_sf7_cross_instance
                                network: end2end
                       groundtruth_code: 35
                ratio_bt_train_and_test: 0.8
                         checkpoint_dir: ./evaluations/v0_checkpoints
                            dir_comment: v0
                             sample_dir: ./evaluations/v0_samples
                                   load: 1
                               log_step: 1000
                           sample_every: 10000
                       checkpoint_every: 5000
                              n_classes: 128
                              stft_nfft: 1024
                            stft_window: 64
                           stft_overlap: 32
                          conv_dim_lstm: 1024
                              freq_size: 128
                       evaluations_path: ./evaluations
================================================================================
length of training and testing data is 10959,2740
Models moved to GPU.
Testing Iteration [    0/14043]
Testing Iteration [ 1000/14043]
Testing Iteration [ 2000/14043]
Testing Iteration [ 3000/14043]
Testing Iteration [ 4000/14043]
Testing Iteration [ 5000/14043]
Testing Iteration [ 6000/14043]
Testing Iteration [ 7000/14043]
Testing Iteration [ 8000/14043]
Testing Iteration [ 9000/14043]
Testing Iteration [10000/14043]
Testing Iteration [11000/14043]
Testing Iteration [12000/14043]
Testing Iteration [13000/14043]
Testing Iteration [14000/14043]
Accuracy:
-25 0.1686131386861314
-24 0.2029197080291971
-23 0.2821167883211679
-22 0.3627737226277372
-21 0.48905109489051096
-20 0.6145985401459854
-19 0.745985401459854
-18 0.8503649635036497
-17 0.9065693430656935
-16 0.9525547445255474
-15 0.964963503649635
-14 0.9744525547445255
-13 0.9824817518248176
-12 0.9890510948905109
-11 0.9875912408759124
-10 0.9927007299270073
-9 0.9934306569343065
-8 0.9919708029197081
-7 0.9923357664233576
-6 0.9945255474452555
-5 0.9927007299270073
-4 0.9937956204379562
-3 0.9952554744525547
-2 0.9963503649635036
-1 0.9952554744525547
0 0.9952554744525547
1 0.9956204379562044
2 0.9967153284671533
3 0.9963503649635036
4 0.9967153284671533
5 0.997080291970803
6 0.995985401459854
7 0.997080291970803
8 0.9985401459854014
9 0.9956204379562044
10 0.9945255474452555
11 0.995985401459854
12 0.997080291970803
13 0.9978102189781022
14 0.995985401459854
15 0.9963503649635036
```

### Evaluation ###

1. Preparation:
  - Get baseline performance:
    You can either download it from [here](https://drive.google.com/drive/folders/1iODrhHg6DmSuAGlq5eTSKClybjTLs9ot?usp=sharing), or run a baseline decode locally with our provided script:
    `matlab/generate_baseline.m`.
    BEFORE RUNNING point the raw_data_dir path in `matlab/generate_baseline.m` to /path/to/raw_sf7_cross_instance
    Note that the result of abs accumulation decode method and phase compensation method reach similar decode accuracy. You can validate it with changing the `abs_decode` in our `generate_baseline.m` script. 
  - Get NELoRa performance:
  - Copy your decode result from `pytorch/*.mat` to `matlab/evaluation/`
2. Run `matlab/evaluation.m` to plot the two results
3. You will get the Symbol Error Rate (SER)result:
![](./matlab/res/result.png)


### From the Scratch ###
To train a model from scratch, we recommand you have a NVIDIA video card, and support cuda acceleration.

1. Set the root path, data path, and evaluation_dir as above. 

2. run the following command:
```
python main.py --data_dir /path/to/raw_sf7_cross_instance --normalization --train_iter 0 --ratio_bt_train_and_test 0.8 
```
The SNR range used in training is specified in '--snr_list' parameter. When training from scratch it is recommended to first train the model at a high SNR (e.g. 0), then gradually descend the SNR.

3. Check your loss with the std print. e.g.:
   - __Iteration [ 1000/100000] | G_Y_loss: 5.5639| G_Image_loss: 2.6935| G_Class_loss: 2.8704__
   - G_Y_loss: G_Image_loss + G_Class_loss
   - G_Image_loss: The loss between groundtruth chirp and your denoised chirp.
   - G_Class_loss: The loss of decoding correct symbol compares with the {Code Label}
NOTE: these are losses on the training dataset. Testing is only done after all training steps are completed (--train_iters).

4. Check your samples in [evaluations_dir]:

Example, a chirp code 24 under **-23** dB noise.

<!-- <p float="left">
  <img src="./imgs/1_-23_7_125000_6_24.1_24_8_raw_0.png" width="100" />
  <img src="/img2.png" width="100" />
  <img src="/img3.png" width="100" />
</p> -->
|   Raw  Chirp  |  GroundTruth |  Denoised |
|:-------------:|:------------:|:---------:|
| ![](./matlab/res/1_-23_7_125000_6_24.1_24_8_raw_0.png)  |  ![](./matlab/res/1_-23_7_125000_6_24.1_24_8_groundtruth_0.png) | ![](./matlab/res/1_-23_7_125000_6_24.1_24_8_fake_0.png) |


5. Get a decode result with in your `pytorch/`. Named as:
[dir_comment]\_[sf]\_[bw].mat (e.g., sf7_v1_7_125000.mat)


