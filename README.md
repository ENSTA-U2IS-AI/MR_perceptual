# MR_perceptual
A Study of Deep Perceptual Metrics for Image quality Assessment. This is the implementation of the paper "A study of deep perceptual metrics for image quality assessment" available [here](https://arxiv.org/pdf/2202.08692.pdf).

## Overview

This repository is the implementation of the MR_Perceptual loss. This code allows the user to compute the MR_Perceptual loss on the 2AFC dataset and custom images. 

![Alt text](imgs/schema.png?raw=true "Title")

## Installation

### Requirements

This code is tested on python 3.8 the useful packages are listed in `requirements.txt`

`pip install -r requirements.txt`

### Datasets

Run `bash ./scripts/download_dataset.sh` to download and unzip the dataset into directory `./dataset`. It takes [6.6 GB] total. Alternatively, run `bash ./scripts/download_dataset_valonly.sh` to only download the validation set [1.3 GB].
- 2AFC train [5.3 GB]
- 2AFC val [1.1 GB]
- JND val [0.2 GB]  

## Basic Usage 

### Reproduce the results

The script `test_dataset_model.py` performs the metric on the 2AFC dataset :

To reproduce the results of the MR_perceptual metric, please use the parameter --mrpl

`python test_dataset_model.py --mrpl`

It is also possible to compute the MR_perceptual metric in various setups. For example, to perform the metric by using cross-entropy loss, sigmoid, linear features, and x1 resolution, run :

`python test_dataset_model.py --loss CE --norm sigmoid --feature linear --resolution x1`

### Use the MRPL metric

The script `test_network.py` can be used to compute the MR_perceptual metric between two given images compared to a reference image. Two setups are available :
- mrpl, which computes the MR_perceptual loss
- mr_simple, which computes the MR_perceptual loss without using gramm matrix

For example, if you want to compute the MR_perceptual loss between two images, please run :

`python test_network.py --ref PATH/TO/REFERENCE_IMG --img1 PATH/TO/IMG_1  --ref PATH/TO/IMG_2 --mode mrpl`

## pip package

For the users which want to perform easily the metric, a pip package is available. For this, run :

`pip install MR_perceptual`

### Usage Example

Here an usage example to perform the metric : 

```
from MR_perceptual import test_network

dist = test_network.MRPL_2_images(PATH/TO/IMG_1,PATH/TO/REFERENCE_IMG)

print(dist)
```

## Acknowledgements

This repository is based on the [PerceptualSimilarity] repo (https://github.com/richzhang/PerceptualSimilarity) Thanks to the authors! 
