# -*- coding: utf-8 -*-
"""
@author: Shuying
copyright: Quing Zhu, Optical and Ultrasound Imaging Laboratory
Email: zhu.q@wustl.edu

This code is used for training and validation using simulation data. 
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from models import TarToPert

# %%
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def preprocess(tar,pert,NoiseSTD,test_ratio):
    assert not np.any(np.isnan(tar.iloc[:,:252]))
    assert not np.any(np.isnan(pert.iloc[:,:252]))

    tarNoise = tar.copy()
    # make the max amplitude the same among all data
    tarNoise.iloc[:,0:126] = tarNoise.iloc[:,0:126].sub(tarNoise.iloc[:,0:126].max(axis=1), axis=0) - 1
    # make the min phase the same among all data 
    tarNoise.iloc[:,126:252] = (tarNoise.iloc[:,126:252].sub(tarNoise.iloc[:,126:252].min(axis=1), axis=0)) + 1         
    # add Gaussian noise        
    NoiseLevel = np.random.normal(1, NoiseSTD, [tar.shape[0],252])
    tarNoise.iloc[:,0:252]= tarNoise.iloc[:,0:252] * NoiseLevel
    
    # train/validation split
    X_train, X_val, y_train, y_val = train_test_split(tarNoise, pert, test_size=test_ratio, random_state=seed)
 
    X_train = X_train.iloc[:,0:252]
    X_val = X_val.iloc[:,0:252]
    y_train = y_train.iloc[:,0:252]
    y_val = y_val.iloc[:,0:252]
      
    X_train = torch.Tensor(X_train.to_numpy()) 
    y_train = torch.Tensor(y_train.to_numpy()) 
    X_val = torch.Tensor(X_val.to_numpy()) 
    y_val = torch.Tensor(y_val.to_numpy()) 
    return X_train, y_train, X_val, y_val
# %%
def main():    
    ########################  data needs to be replaced by users' own data  ############################
    # input: target log amplitude and phase;
    tar = pd.read_csv('data/tar.csv')
    # output: real and imaginary part of perturbation
    pert = pd.read_csv('data/pert.csv')
        
    ########### loop over parameters (the ones used in the manuscript are provided here) ################
    for NoiseSTD in [0.02]: 
        test_ratio = 0.2 
        X_train, y_train, X_val, y_val = preprocess(tar,pert,NoiseSTD,test_ratio)
        for num_epochs in [200]: 
            for batch_size in [64]:     
                train = torch.utils.data.TensorDataset(X_train, y_train)
                trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)    
                test = torch.utils.data.TensorDataset(X_val, y_val)
                testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)
                for learning_rate in [1e-4]:                
                    for reg in [1e-5]: 
                        
                        train_losses = [] # to track the training loss as the model trains
                        valid_losses = [] # to track the validation loss as the model trains                            
                        avg_train_losses = [] # to track the average training loss per epoch as the model trains      
                        avg_valid_losses = [] # to track the average validation loss per epoch as the model trains

                        model = TarToPert() # model may need adjustment for different DOT systems
                        model.to(device)
                        criterion = nn.MSELoss()
                        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)
                        # reduce lr if loss does not decrease 
                        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3) 
                        
                        for epoch in range(num_epochs):
                            # ===================training=====================
                            for data, label in trainloader:
                                data = data.to(device)  
                                label = label.to(device)  
                                # ===================forward=====================
                                output = model(data)
                                loss = criterion(output, label)
                                train_lossi = loss.data.item()
                                train_losses.append(train_lossi)
                                # ===================backward====================
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()                                    
                            # ===================validation=====================
                            model.eval()     
                            for data, labels in testloader:
                                data = data.to(device)
                                labels = labels.to(device)
                                output = model(data)        
                                loss = criterion(output,labels)
                                val_lossi = loss.data.item()
                                valid_losses.append(val_lossi)
                            # ===================log========================
                            train_loss = np.average(train_losses)
                            val_loss = np.average(valid_losses)
                            avg_train_losses.append(train_loss)
                            avg_valid_losses.append(val_loss)
                            
                            print('epoch [{}/{}], training loss:{:.4f}'.format(epoch + 1, num_epochs, train_loss) +
                                    ', validation loss:{:.4f}'.format(val_loss))  
                            
                            scheduler.step(val_loss)

                        torch.save(model, 
                                    'TarToPert.pth'
                                    )
    # plot losses to make sure no overfitting
    plt.figure
    plt.plot(avg_train_losses)
    plt.plot(avg_valid_losses)
    plt.legend(['train loss', 'validtion loss'])
    plt.show()        

if __name__ == "__main__":
   main()
