import pandas as pd
import numpy as np
import cv2
from skimage.io import imread
import matplotlib.pyplot as plt
import json
import pickle
import os
import glob
import random
import nibabel as nib
import imageio
from IPython.display import Image
from sklearn.preprocessing import MinMaxScaler
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim   
from torchvision import datasets , models  
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms 
from torch.utils.data import DataLoader , Dataset
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid , save_image , draw_segmentation_masks
!pip install torchsummary
from torchsummary import summary



class DoubleConv(nn.Module):
    def __init__(self , in_channels , out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels , out_channels , kernel_size=3 , stride=1 , padding=1 , bias=False),
            nn.BatchNorm3d(out_channels),    # try instance normalization
            nn.LeakyReLU(negative_slope=0.2 , inplace=True),
            nn.Conv3d(out_channels , out_channels , kernel_size=3 , stride=1 , padding=1 , bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.2 , inplace=True), 
        )
        
    def forward(self , x):
        return self.conv(x)
    


class EncoderBlock(nn.Module):
    def __init__(self , in_channels , out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels , out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2 , stride=2)
        
    def forward(self , t):
        t = self.conv(t)
        p = self.pool(t)
        return t , p
    
    
    
class DecoderBlock(nn.Module):
    def __init__(self , in_channels , out_channels):   # 256, 128
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels , out_channels , kernel_size=2 , stride=2 , padding=0)
        self.conv = DoubleConv(out_channels*2 , out_channels)
        
    def forward(self , x , skip):
        x = self.up(x)
        x = torch.cat([x , skip] , axis = 1)
        x = self.conv(x)
        return x
    

    
class UNET_3D(nn.Module):
    def __init__(self , in_channels = 3 , out_channels = 3 , features = [16, 32, 64, 128]):   # 16, 32, 64, 128
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
        self.clf = nn.Conv3d(features[0] , 3 , kernel_size=1 , stride=1 , padding=0)
        
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
    x = torch.randn(shape).to('cuda')
    
    model = UNET()
    model.to('cuda')
    preds = model(x)
    
    print(preds.shape)
    assert preds.shape == (2, 3, 128, 128, 128)

    
test((2, 3, 128, 128, 128))



x = torch.randn((2, 3, 128, 128, 128)).to('cuda')
    
model = UNET()
model.to('cuda')

summary(model, input_size = (3, 128, 128, 128), batch_size = 5)

