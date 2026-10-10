#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 14:37:20 2022

@author: akila
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
class SimpleCNN_v2(torch.nn.Module):

    def __init__(
        self,
        n_classes = 11,
        dropout =  0.5,
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super(SimpleCNN_v2, self).__init__()
        print('here')
        self.device = device
        self.hidden_size = 50
        self.seq_length = 5     
        self.num_lstm_layers  = 1
                
        self.pad1 =  nn.ZeroPad2d((2, 2, 0, 0,))
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(1, 3), stride=1, padding=0)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.cnn2 = nn.Conv2d(in_channels=256, out_channels=80, kernel_size=(2, 3), stride=1, padding=0)
        self.fl = nn.Flatten()
        self.fc1 = nn.Linear(in_features=self.hidden_size, out_features=256, bias=True)
        self.fc2 =  nn.Linear(in_features=256, out_features=n_classes, bias=True)
        self.lstm = nn.LSTM(1320, hidden_size=self.hidden_size, num_layers=self.num_lstm_layers)

# =============================================================================
#         self.model = nn.Sequential(
#             nn.ZeroPad2d(padding=(2, 2, 0, 0,)),    # zero pad front/back of each signal by 2
#             nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(1, 3), stride=1, padding=0),
#             nn.ReLU(),
#             nn.Dropout(p=dropout),
#             nn.ZeroPad2d(padding=(2, 2, 0, 0,)),    # zero pad front/back of each signal by 2
#             nn.Conv2d(in_channels=256, out_channels=80, kernel_size=(2, 3), stride=1, padding=0),
#             nn.ReLU(),
#             nn.Dropout(p=dropout),
#             nn.Flatten(),
#             nn.Linear(in_features=10560, out_features=256, bias=True),
#             nn.ReLU(),
#             nn.Dropout(p=dropout),
#             nn.Linear(in_features=256, out_features=n_classes, bias=True),)
# =============================================================================
        

    def forward(self, x):
        x = self.dropout(self.relu(self.cnn2(self.pad1(self.dropout(self.relu(self.cnn1(self.pad1(x))))))))        
        x = x.view(-1, 8,10*132)
        h_0 = Variable(torch.zeros(self.num_lstm_layers, x.size(1), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_lstm_layers, x.size(1), self.hidden_size))
        
        output, (hn,cn) = self.lstm(x, (h_0, c_0))
        out = output[:,-1,:]
        x = self.fc1(out)
        x = self.fc2(self.dropout(self.relu(self.fc1(out))))
      
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

