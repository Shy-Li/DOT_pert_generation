# -*- coding: utf-8 -*-
"""
copyright: Quing Zhu, Optical and Ultrasound Imaging Laboratory
Email: zhu.q@wustl.edu

This code is for generating perturbation for testing set (phantom/patient data). Our phantom data is provided here. 
"""
import torch
import numpy as np
import pandas as pd
from models import TarToPert

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = TarToPert()
model = torch.load('data/TarToPert.pth',map_location=device)

# Input the pre-processed phantom/patient data with max log amplitude and min phase fixed 
X_test = pd.read_csv('data/phantom_tar1.csv') 
X_test = X_test.values[:,:252]
X_test = torch.Tensor(X_test) 
X_test = X_test.to(device) 
if  torch.cuda.is_available():
    y_pred = model(X_test).cpu().detach().numpy()
else:
    y_pred = model(X_test).detach().numpy()

pd.DataFrame(y_pred).to_csv('results/pert_pred_phantom_1.csv', index = False)