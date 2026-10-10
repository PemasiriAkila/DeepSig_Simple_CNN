#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 14:37:20 2022

@author: akila
"""
import torch
import torch.nn as nn
class SimpleCNN(torch.nn.Module):

    def __init__(
        self,
        n_classes: int = 24,
        dropout: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super(SimpleCNN, self).__init__()
        print('here')
        self.device = device
        
        self.pad1 =  nn.ZeroPad2d((2, 2, 0, 0,))
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(1, 3), stride=1, padding=0)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.cnn2 = nn.Conv2d(in_channels=256, out_channels=80, kernel_size=(2, 3), stride=1, padding=0)
        self.fl = nn.Flatten()
        self.fc1 = nn.Linear(in_features=82240, out_features=256, bias=True)
        self.fc2 =  nn.Linear(in_features=256, out_features=n_classes, bias=True)
        
        

        

    def forward(self, x):
        x = self.fl(self.dropout(self.relu(self.cnn2(self.pad1(self.dropout(self.relu(self.cnn1(self.pad1(x)))))))))
        x = self.fc2(self.dropout(self.relu(self.fc1(x))))
        return x
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        x = x.to(self.device)
        y_pred = self.model(x)
        y_pred = y_pred.to("cpu")
        y_pred = torch.softmax(y_pred, dim=-1)
        values, indices = torch.max(y_pred, dim=-1)
        indices = indices.numpy()
        return indices

