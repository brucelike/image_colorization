# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 13:43:15 2021

@author: bruce
"""
import torch
from basic_model import Net
from PIL import Image
import torchvision.transforms as T

model_path='trained.pth'
class infer:
    def __init__(self, model_path):
        self.model=Net()
        self.pretrained=torch.load(model_path)
        self.model.load_state_dict(self.pretrained)
    
    
    
    
    
if __name__ == '__main__':
    #model=basic_model.Net()


        # Use the input transform to convert images to grayscale
        input_transform = T.Compose([T.ToTensor(),
                                          T.Resize(size=(256,256)),
                                          T.Grayscale(),
                                          T.Normalize((0.5), (0.5))
                                          ])
        # Use this on target images(colorful ones)
        target_transform = T.Compose([T.ToTensor(),
                                           T.Resize(size=(256,256)),
                                           T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])