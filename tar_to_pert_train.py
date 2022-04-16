# -*- coding: utf-8 -*-
"""
Created on Sat May 30 11:02:22 2020

@author: Shuying
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
    
    X_train, X_test, y_train, y_test = train_test_split(tarNoise, pert, test_size=test_ratio, random_state=seed)
 
    X_train = X_train.iloc[:,0:252]
    X_test = X_test.iloc[:,0:252]
    y_train = y_train.iloc[:,0:252]
    y_test = y_test.iloc[:,0:252]
      
    X_train = torch.Tensor(X_train.to_numpy()) 
    y_train = torch.Tensor(y_train.to_numpy()) 
    X_test = torch.Tensor(X_test.to_numpy()) 
    y_test = torch.Tensor(y_test.to_numpy()) 
    return X_train, y_train, X_test, y_test
# %%
def main():
    
    ########################  data needs to be replaced by users' own data  ############################
    # input: target log amplitude and phase;
    tar = pd.concat([ 
                     pd.read_csv('data/data_tar1.csv'),
                     pd.read_csv('data/data_tar2.csv'),
                     pd.read_csv('data/tar_MC.csv'),
                     pd.read_csv('data/tar_irreg.csv'),
                     pd.read_csv('data/data_tar_shallow.csv'),
                     ])
    # output: real and imaginary part of perturbation
    pert = pd.concat([
                      pd.read_csv('data/pert1.csv'),
                      pd.read_csv('data/pert2.csv'),
                      pd.read_csv('data/pert_MC.csv'),
                      pd.read_csv('data/pert_irreg.csv'),
                      pd.read_csv('data/pert_shallow.csv'),
                      ])
    test_ratio = 0.2 
    
    ########### loop over parameters (the ones used in the manuscript are provided here) ################
    for NoiseSTD in [0.02]: 
        X_train, y_train, X_test, y_test = preprocess(tar,pert,NoiseSTD,test_ratio)
        for num_epochs in [200]: 
            for batch_size in [64]:     
                train = torch.utils.data.TensorDataset(X_train, y_train)
                trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)    
                test = torch.utils.data.TensorDataset(X_test, y_test)
                testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)
                for learning_rate in [1e-4]: 
                    for neck in [128]:
                        for reg in [1e-5]: 
                            
                            train_losses = [] # to track the training loss as the model trains
                            valid_losses = [] # to track the validation loss as the model trains                            
                            avg_train_losses = [] # to track the average training loss per epoch as the model trains                            
                            avg_valid_losses = [] # to track the average validation loss per epoch as the model trains

                            model = TarToPert(neck)
                            model.to(device)
                            criterion = nn.MSELoss()
                            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)
                            
                            for epoch in range(num_epochs):
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
                                
                                scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)
                                scheduler.step(val_loss)

                            torch.save(model, 
                                       'TarToPert_epoch' + str(epoch+1) + '_bz' +  str(batch_size) 
                                       + '_lr' +  str(learning_rate)+'_neck' +  str(neck) + '_reg' + str(reg)
                                       + '_noise_' + str(NoiseSTD * 100) + '.pth'
                                       )
                   
        plt.figure
        plt.plot(avg_train_losses)
        plt.plot(avg_valid_losses)
        plt.legend(['train loss', 'validtion loss'])
        plt.show()        

if __name__ == "__main__":
   main()
