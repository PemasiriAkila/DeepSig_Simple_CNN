#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 14:18:36 2022

@author: akila
"""
import torch
#from tqdm import tqdm
from typing import Tuple, List, Dict
import numpy as np
import gc
import pickle
from torch.utils.data import Dataset
import itertools
import scipy.io as sio
import h5py
from torch.utils.data import DataLoader
class RadioML_2018(Dataset):
   




    def __init__(
        self,
        data_dir: str = "./data",
        #file_name: str = "/home/n9546481/RF-Project/RadioML_data/RadioML_2016.mat"
        #file_name: str = "/home/hewathon/RF-Project_new/RadComDataset/RadComAWGN.mat"
        #file_name: str = "/home/akila/Documents/RF-Project/Using_DeepsigData/DeepSig_2018/2018.01.OSC.0001_1024x2M.h5/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5"
       
    ):
        self.data = self.__get_data_from_h5py('/home/akila/Documents/RF-Project/Using_DeepsigData/DeepSig_2018/2018.01.OSC.0001_1024x2M.h5/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5')
        
        #self.data = self.__get_data_from_h5py('/home/hewathon/RF-Project_new/Deepsig_2018_dataset/Segmented_data/GOLD_XYZ_OSC.0001_1024.hdf5')



    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

            signal = self.data['signals'][idx]
            mod_type = np.argmax(self.data['labels'][idx])
            SnR = self.data['SNRs'][idx]
            

            return signal, mod_type, SnR

    def __len__(self) -> int:
        return len(self.data['signals'])
    
    def __get_data_from_h5py(self, file_name):
        File = h5py.File(file_name)
        labels = File[list(File.keys())[1]]
        print(labels.shape)

        signals = File[list(File.keys())[0]]
        print(signals.shape)
        SNRs = File[list(File.keys())[2]]
        print(SNRs.shape)
        
        return dict(file=File, signals= signals, SNRs = SNRs, labels = labels)
    
