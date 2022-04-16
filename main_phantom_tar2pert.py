# -*- coding: utf-8 -*-
"""
Created on Tue May 25 17:18:57 2021
copyright: Quing Zhu, Optical and Ultrasound Imaging Laboratory
Email: zhu.q@wustl.edu

This code is for generating perturbation for phantom/patient data. 
"""
import torch
import numpy as np
import pandas as pd
from models import TarToPert

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load('TarToPert_epoch200_bz64_lr0.0001_neck128_reg1e-05_noise_2.0.pth').to(device)

# Input the pre-processed phantom/patient data with max log amplitude and min phase fixed 
X_test = pd.read_csv('phantom_tar1.csv') 
X_test = X_test.values[:,:252]
X_test = torch.Tensor(X_test) 
X_test = X_test.to(device)  
y_pred = model(X_test).cpu().detach().numpy()
# pd.DataFrame(y_pred).to_csv('pert_pred_phantom.csv', index = False)