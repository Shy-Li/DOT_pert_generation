# DOT_pert_generation
A MLP model to generate data for DOT difference imaging from target measurements only. 

## Table of Contents

- [Background](#background)
- [Install](#install)
- [MLP model](#model)
- [Training](#training)
- [Pre-trained model](#pretrained)
- [Testing](#testing)
- [Citation](#citation)

## Background
### Significance: “Difference imaging”, which reconstructs target optical properties using measurements with and without target information, is often used in diffuse optical tomography (DOT) in vivo imaging. However, taking additional reference measurements is time-consuming, and mismatches between the target medium and the reference medium can cause inaccurate reconstruction. 
### Aim: We aim to simplify the data acquisition and mitigate the mismatch problems in DOT difference imaging by using a deep learning-based approach to generate data from target measurements only. 

## Install
The code was tested with Python 3.7.11.

Required packages: 
 - pytorch 1.8.1
 - numpy 1.12.1
 - pandas 1.1.3
 - matplotlib 3.4.3
 - scikit-learn 0.24.2
 
## Model 
The MLP model is the class `TarToPert` in `models.py`. The bottleneck is tunable by changing the parameter "neck". 

## Training
The training & validation is done by `tar_to_pert_train.py`. The users need to replace the dataset with their own. 

## Pretrained
The pre-trained model is saved in `TarToPert_epoch200_bz64_lr0.0001_neck128_reg1e-05_noise_2.0.pth`.

## Testing
A sample testing code is given in `main_phantom_tar2pert.py`. 

## Citation
