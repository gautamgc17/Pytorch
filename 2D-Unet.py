#!/usr/bin/env python
# coding: utf-8

# IMPORTS
import torch
import torchvision         
import torch.nn as nn           
import torch.optim as optim          
import torch.nn.functional as F   
from torch.utils.data import DataLoader , Dataset   
from torchvision import datasets , models            
import torchvision.transforms as transforms  
from torchsummary import summary
from torchvision.utils import make_grid , save_image
from tqdm import tqdm 
import numpy as np
import random

seed = 42
torch.manual_seed(seed)
np.random.seed(42)



class DoubleConv(nn.Module):
    def __init__(self , in_channels , out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels , out_channels , kernel_size=3 , stride=1 , padding=1 , bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels , out_channels , kernel_size=3 , stride=1 , padding=1 , bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), 
        )
        
    def forward(self , x):
        return self.conv(x)



class EncoderBlock(nn.Module):
    def __init__(self , in_channels , out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels , out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2 , stride=2)
        
    def forward(self , t):
        t = self.conv(t)
        p = self.pool(t)
        return t , p



class DecoderBlock(nn.Module):
    def __init__(self , in_channels , out_channels):   # 256, 128
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels , out_channels , kernel_size=2 , stride=2 , padding=0)
        self.conv = DoubleConv(out_channels*2 , out_channels)
        
    def forward(self , x , skip):
        x = self.up(x)
        x = torch.cat([x , skip] , axis = 1)
        x = self.conv(x)
        return x    



class UNET(nn.Module):
    def __init__(self , in_channels = 3 , out_channels = 1 , features = [16, 32, 64, 128]):   # 16, 32, 64, 128
        super().__init__() 
        
        """ Encoder Path """
        self.encoder_path = nn.ModuleList()
        for feature in features:
            self.encoder_path.append(EncoderBlock(in_channels , feature))
            in_channels = feature
    
        """Bottle Neck"""
        self.b = DoubleConv(features[-1] , features[-1]*2)
        
        """Decoder Path"""
        self.decoder_path = nn.ModuleList()
        for feature in reversed(features):
            self.decoder_path.append(DecoderBlock(feature*2 , feature))
            
        """Classifier"""
        self.clf = nn.Conv2d(features[0] , 1 , kernel_size=1 , stride=1 , padding=0)
        
    def forward(self , x):
        c1 , p1 = self.encoder_path[0](x)   # 16, 128, 128 : 16, 64, 64
        c2 , p2 = self.encoder_path[1](p1)  # 32, 64, 64 ; 32, 32, 32
        c3 , p3 = self.encoder_path[2](p2)  # 64, 32, 32 ; 64, 16, 16
        c4 , p4 = self.encoder_path[3](p3)  # 128, 16, 16 ; 128, 8, 8
        
        b = self.b(p4)   #256, 8, 8
        
        d1 = self.decoder_path[0](b, c4)     
        d2 = self.decoder_path[1](d1, c3)
        d3 = self.decoder_path[2](d2, c2)
        d4 = self.decoder_path[3](d3, c1)
        
        output = self.clf(d4)
        return output



def test(shape):               # pass a tuple
    x = torch.randn(shape)
    
    model = UNET()
    preds = model(x)
    
    assert preds.shape == (shape[0], 1, 128, 128)



test((16, 3, 128, 128))


test((64, 3, 128, 128))
