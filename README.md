# U-Net Human Background Remover 

This project demonstrates the implementation of a U-Net model for background removal in images. The model is trained on the AISegment dataset to accurately segment the foreground from the background in images, specifically human matting.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Acknowledgements](#acknowledgements)
- [References](#references)

## Introduction
The U-Net model is a type of convolutional neural network (CNN) designed for image segmentation tasks. This project focuses on using U-Net to remove backgrounds from images, leaving only the desired foreground object.

## Requirements
- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- OpenCV
- PIL

## Dataset
The dataset used for this project is the [AISegment Matting Human Datasets](https://www.kaggle.com/datasets/laurentmih/aisegmentcom-matting-human-datasets). It consists of images and their corresponding matting masks.

## Usage
1. Clone the repository.
2. Install the required packages.
3. Download the dataset from Kaggle and extract it.
4. Run the provided code to train and evaluate the model.

## Model Architecture
The U-Net model consists of an encoder (contracting path) and a decoder (expansive path) with skip connections between corresponding layers. This architecture helps in capturing both spatial and contextual information effectively.

## Training
The model is trained using the binary cross-entropy loss function and the Adam optimizer. Early stopping and model checkpointing prevent overfitting and save the best model.

## Evaluation
The model is evaluated using accuracy score. Visualizations of the input images, ground truth masks, and predicted masks are provided to assess the model's performance.

## Results
The results include the visualizations of the input images, ground truth masks, predicted masks, and the final segmented output. The learning curve of the model is also plotted to show the training progress.

## Acknowledgements
- The dataset is provided by AISegment on Kaggle.
- The U-Net model is inspired by the original paper [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).

## References
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [AISegment Matting Human Datasets](https://www.kaggle.com/datasets/laurentmih/aisegmentcom-matting-human-datasets)
