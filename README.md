## Image Segmentation

This repository contains codes for Kaggle competition of ”Finding and Measuring Lungs in CT Data”:
https://www.kaggle.com/datasets/kmader/finding-lungs-in-ct-data

The competition goal is processing and trying to find lesions in CT images of the lungs. In fact, in order to find disease in these images well, it is important to first find the lungs well. 

## Data
The dataset is A collection of CT images, manually segmented lungs and measurements in 2/3D.
The data source is:
https://www.kaggle.com/datasets/kmader/finding-lungs-in-ct-data

## Architecture
We use two different architectures for training the model in this package.
1. Shallow CNN
2. U-Net (Convolutional Networks for Biomedical Image Segmentation).
The UNet architecture is illustrated in the following:
<p align="center">
    <img src="https://github.com/rezatorabi13/Image_Segmentation/blob/main/docs/Unet-architecture.png" alt="Figure1" width="600"/>
    <br>
    <em>The UNet architecture.</em>
</p>

## Training result
An example of the model performance is shown based of **dice coefficient**.
<p align="center">
    <img src="https://github.com/rezatorabi13/Image_Segmentation/blob/main/docs/CT_Dice_coeficient.jpg" alt="Figure2" width="400"/>
    <br>
    <em>Dice coefficient versus epoch.</em>
</p>

## Prediction Results
Some samples of prediction results comparing to the original masks:
<p align="center">
    <img src="https://github.com/rezatorabi13/Image_Segmentation/blob/main/docs/CT_pred.jpg" alt="Figure3" width="600"/>
    <br>
    <em>prediction results.</em>
</p>

## Instructions for using the Package

1. Download the repository to data_raw directory and unzip it.

2. Choose one of the models during the training.

 For more information, refer to the explanation in each script.

### Requirements
This code requires you have the following libraries installed. 
- Tensorflow
- Keras
- Skimage
- OpenCV (optional)
- numpy
- matplotlib
- glob
- Scipy
