#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 14:32:16 2022

@author: akila
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import gc
import itertools
import pickle
import torch
import numpy as np
#from tqdm import tqdm
#from typing import Tuple, List, Dict
from utils.visualizer import Visualizer
import os.path as osp
import scipy.io as sio
from collections import OrderedDict
from networks import SimpleCNN

from data import Deepsing_Data


from torch.utils.tensorboard import SummaryWriter


curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path,  'cachedir')
print(cache_path)

TRAIN_SPLIT= 0.7
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1
BATCH_SIZE = 512
n_epochs = 8
save_epoch_freq = 1
print_val_scalars= True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


checkpoint_dir = 'simple_v4_from_mat_accuracy'


name = 'simple_epoch'+str(n_epochs)
results_file_location = cache_path+'/'+checkpoint_dir+'/'+name+'/'+'results/'

if not os.path.exists(cache_path):
    os.makedirs(cache_path)


if not os.path.exists(osp.join(cache_path, checkpoint_dir)):
    os.makedirs(osp.join(cache_path, checkpoint_dir))
    

if not os.path.exists(osp.join(cache_path, checkpoint_dir, name)):
    os.makedirs(osp.join(cache_path, checkpoint_dir, name))
    
if not os.path.exists(osp.join(cache_path, checkpoint_dir, name, 'results')):
    os.makedirs(osp.join(cache_path, checkpoint_dir, name, 'results'))


writer = SummaryWriter(cache_path+'/'+ checkpoint_dir+'/'+name)   
 
dataset = Deepsing_Data.RadioML_2018()
visualizer = Visualizer(cache_path,checkpoint_dir, name)
total = len(dataset)
lengths = [int(len(dataset)*TRAIN_SPLIT)]
lengths.append(int(len(dataset)*VAL_SPLIT))
lengths.append(total - lengths[0]-lengths[1])
train_set, val_set, test_set = random_split(dataset, lengths)


# Setup dataloaders
trainloader = DataLoader(train_set, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

model =  SimpleCNN.SimpleCNN()
model =  model.to(device)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer= optim.Adam(model.parameters(), lr = 0.0001)

total_steps = 0
print_scalars = True
print_freq = 10
i = 0
min_val_loss = 10000000
for epoch in range(0,n_epochs):
    print(epoch)
    model =  model.to(device)
    model.train()
    epoch_iter = 0
    tot_epoch_loss_train  = 0
    mod_gt_list = []
    mod_pred_list = []
    

    for data in trainloader:
        signal, mod_type, SnR = data
        pred =  (model(data[0].to(device).view(-1,1,2,1024)))
        pred_vals = torch.tensor(np.argmax(pred.cpu().detach().numpy(), axis=1)).view(-1,1)
        pred_original = pred
        target_original = data[1].to(device)
    
        pred_float = pred_original.float()
        target_float = target_original.float()
        pred_long = pred_original.long()
        target_long = target_original.long().reshape((-1,))
        
        #rint(target_long, pred)

        loss = criterion(pred, target_long)
        
        writer.add_scalar("Loss/train", loss, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mod_pred_val = torch.argmax(pred,dim=1)

        
        mod_gt_list = mod_gt_list+ mod_type.tolist()
        mod_pred_list = mod_pred_list+mod_pred_val.tolist()
        

        
        total_steps += 1
        epoch_iter +=1 
        tot_epoch_loss_train += loss.detach()

        
        if print_scalars and (total_steps % print_freq == 0):
            scalars = OrderedDict([('Cross_entropy_loss', loss.item())])
            visualizer.print_current_scalars(epoch, epoch_iter, scalars)
        i = i+BATCH_SIZE
    correct_mod = (np.array(mod_gt_list) == np.array(mod_pred_list)).sum()
    writer.add_scalar("AccuracyMod/train", 100*correct_mod/len(mod_gt_list), epoch) 
    

    model.train(False)
    print('epoch finished')
    val_loss = 0
    val_iter = 0
    
    mod_gt_list = []
    mod_pred_list = []
    
    for data in val_loader:
        signal, mod_type, SnR = data
        pred =  (model(data[0].to(device).view(-1,1,2,1024)))
        loss= criterion(pred, data[1].to(device).long())
        writer.add_scalar("Loss/val", loss, epoch)
        
        mod_pred_val = torch.argmax(pred,dim=1)

        
        mod_gt_list = mod_gt_list+ data[1].tolist()
        mod_pred_list = mod_pred_list+mod_pred_val.tolist()
        

        val_loss += loss.item() 
        
        if print_val_scalars :
            scalars = OrderedDict([('Cross_entropy_loss', loss.item())])
            visualizer.print_current_scalars_val(epoch, val_iter, scalars)
            
    correct_mod = (np.array(mod_gt_list) == np.array(mod_pred_list)).sum()
    writer.add_scalar("AccuracyMod/val", 100*correct_mod/len(mod_gt_list), epoch) 

    scalars = OrderedDict([('Train_loss_tot', tot_epoch_loss_train), ('Val_loss_tot', val_loss)])
    visualizer.print_scalars_endepoch(epoch, epoch_iter, scalars)


    visualizer.print_val_accuracies(epoch, epoch_iter, scalars)

    if(min_val_loss > val_loss):
        min_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(cache_path,checkpoint_dir, name, 'min_val_'+str(epoch)+'.pth')) 
    if (epoch+1) % save_epoch_freq == 0:
        print('saving the model at the end of epoch {:d}, iters {:d}'.format(epoch, total_steps))
        torch.save(model.state_dict(), os.path.join(cache_path,checkpoint_dir, name, str(epoch)+'.pth')) 

    if (epoch+1) % save_epoch_freq == 0:  
        test_mod_gt = []   
        test_mod_pred = []
        test_SnR = []
        for data in test_loader:
            signal, mod_type, SnR = data
            pred =  (model(data[0].to(device).view(-1,1,2,1024)))
            

            
            mod_pred_val = torch.argmax(pred,dim=1)

            
            test_mod_gt = test_mod_gt+ mod_type.tolist()
            test_mod_pred = test_mod_pred+mod_pred_val.tolist()
            

            
            test_SnR = test_SnR+SnR.tolist()
        
        test_mod_gt = np.array(test_mod_gt)
        test_mod_pred = np.array(test_mod_pred)
        test_SnR= np.array(test_SnR)

        
        save_data = {'gt':test_mod_gt, 'pred':test_mod_pred,'SNR':test_SnR}
        save_file_name = results_file_location+'/'+str(epoch)+'_mod.mat'
        sio.savemat(save_file_name, save_data)
        
        writer.flush()
        
           
        
