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
Significance: “Difference imaging”, which reconstructs target optical properties using measurements with and without target information, is often used in diffuse optical tomography (DOT) in vivo imaging. However, taking additional reference measurements is time-consuming, and mismatches between the target medium and the reference medium can cause inaccurate reconstruction. 
Aim: We aim to simplify the data acquisition and mitigate the mismatch problems in DOT difference imaging by using a deep learning-based approach to generate data from target measurements only. 

## Install
The code was tested with Python 3.7.11.

Required packages: 
 - pytorch 1.8.1
 - numpy 
 - pandas 
 - matplotlib 
 - scikit-learn 
 
## Model 
The MLP model is the class `TarToPert` in `models.py`. The bottleneck is tunable by changing the parameter "neck". The MLP model structures may need to be changed according to uses' DOT system configuration and the size of the dataset. 

## Training
The training & validation is done by `tar_to_pert_train.py`. The users need to use the dataset with their own. 

## Pretrained
The pre-trained model is saved in `TarToPert_pretrained.pth`. 

## Testing
A sample testing code is given in `main_phantom_tar2pert.py`. Sample phantom data was given as `phantom_tar1.csv` and the generated perturbation will be saved as `pert_pred_phantom.csv'. 

## Citation
Li, S., Zhang, M., Xue, M. and Zhu, Q., 2022. Difference imaging from single measurements in diffuse optical tomography: a deep learning approach. Journal of Biomedical Optics, 27(8), p.086003.
