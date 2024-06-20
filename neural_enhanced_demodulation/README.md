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
python main.py --data_dir /path/to/raw_sf7_cross_instance --normalization --train_iter 0 --ratio_bt_train_and_test 0.8 --load --load_iters 100000 --snr -17
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
 --sample_every How many steps for each time a sample of the masked spectrograms are saved to [evaluations_dir]/[dir_comment]_[sample_dir]
 --checkpoint_every How many steps for each time a checkpoint of the model during training is saved to [evaluations_dir]/[dir_comment]_[checkpoint_dir]

3. Get a decode result with in your `pytorch/` directory.

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
python main.py --data_dir /path/to/raw_sf7_cross_instance --normalization --train_iter 0 --ratio_bt_train_and_test 0.8 --snr 0
```
First train the model at a high SNR (e.g. 0), then gradually descend the SNR.
You can also use loops like 
```
for i in `seq 0 25`; do 
    python main.py --data_dir /path/to/raw_sf7_cross_instance --normalization --train_iter 0 --ratio_bt_train_and_test 0.8 --snr -$i;
done
```

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


