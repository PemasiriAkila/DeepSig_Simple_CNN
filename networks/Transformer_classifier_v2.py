#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:01:49 2022

@author: akila
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.nn.functional as f
#class SimpleClassifierNet(nn.Module):
    
#    def __init__(self, out_l = 1, in_l = 1):
#        super(SimpleClassifierNet(), self).__init__()
#        self.cnn1 = nn.Conv2d(in_channels=1,out_channels=out_l,kernel_size=2, stride=1, padding=0)
        
#    def forward(self, x):
#       print('b')
#        return 1
#        #return nn.functional.pad(x, (1,3), "constant", 0)

class SimpleRegressorNet(nn.Module):
    def __init__(self, in_size=1):
        super(SimpleRegressorNet, self).__init__()
        #self.cnn1 = nn.Conv2d(in_channels=1,out_channels=256,kernel_size=(1,3), stride=1, padding=(0,2))
        self.cnn1 = nn.Conv2d(1, 256, [1,3], padding=(0,2))
        self.cnn2 = nn.Conv2d(256, 80, [2,3], padding=(0,2))
        self.fc1 = nn.Linear(41280, 256)
        self.fc2 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.5)
        
        #padding=(0,2))
        #self.linear = nn.Linear(in_size,1,padding=(0,2))
        
    def forward(self,x):
        x = F.relu(self.cnn1(x))
        x = self.dropout(x)
        x = F.relu(self.cnn2(x))
        x = self.dropout(x)
        x = x.view(-1, 80*516)
        x = self.fc2(self.dropout(self.fc1(x)))
        return x

def feed_forward(dim_input: int = 512, dim_feedforward: int = 2048) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_input),
    )


def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    temp = query.bmm(key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    softmax = f.softmax(temp / scale, dim=-1)
    return softmax.bmm(value)

    
class AttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return self.linear(
            torch.cat([h(query, key, value) for h in self.heads], dim=-1)
        )


class Residual(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: Tensor) -> Tensor:
        # Assume that the "query" tensor is given first, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        return self.norm(tensors[0] + self.dropout(self.sublayer(*tensors)))
    
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int = 512,
        num_heads: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        #print('in transformer encoder layer')
        dim_q = dim_k = max(dim_model // num_heads, 1)
        #print('dim_model', dim_model, 'num_heads', num_heads, 'dim q', dim_q, 'dim k', dim_k)
        self.attention = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, src: Tensor) -> Tensor:
        src = self.attention(src, src, src)
        return self.feed_forward(src)
    
def position_encoding(
    seq_len: int, dim_model: int, device: torch.device = torch.device("cpu"),
) -> Tensor:
    #print('in postion encoding')
    #print('seq len in positional encoding', seq_len)
    #print('sdim in positional encoding', dim_model)
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / (1e4 ** (dim // dim_model))
    positional_encoding_out = torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))
    #print('resultant positional encoding', positional_encoding_out.shape)

    return positional_encoding_out

    
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        dim_model: int = 128,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, src: Tensor) -> Tensor:
        seq_len, dimension = src.size(1), src.size(2)
        pe = position_encoding(seq_len, dimension)
        
        
        #src += position_encoding(seq_len, dimension)
        for layer in self.layers:
            src = layer(src)
        #print(src.shape)
        return src
    
    
class TransformerClassifier(nn.Module):
    def __init__(self,         num_encoder_layers: int = 6,
            num_decoder_layers: int = 6,
            dim_model: int = 128, 
            num_heads: int = 6, 
            dim_feedforward: int = 2048, 
            dropout: float = 0.1, ):
        super(TransformerClassifier, self).__init__()
        
        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.pad1 =  nn.ZeroPad2d((2, 2, 0, 0,))
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(1, 3), stride=1, padding=0)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.cnn2 = nn.Conv2d(in_channels=256, out_channels=80, kernel_size=(2, 3), stride=1, padding=0)
        self.fl = nn.Flatten()
        self.fc1 = nn.Linear(in_features=10560, out_features=256, bias=True)
        self.fc2 =  nn.Linear(in_features=256, out_features=11, bias=True)
        
    def forward(self, src: Tensor) -> Tensor:
        encoded_signal = self.encoder(src).view(-1 ,1,2,128)
        x = encoded_signal
        x = self.fc2(self.dropout(self.relu(self.fc1(self.fl(self.dropout(self.relu(self.cnn2(self.pad1(self.dropout(self.relu(self.cnn1(self.pad1(x)))))))))))))
        
        return x
        
        
        