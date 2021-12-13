# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 13:43:15 2021

@author: bruce
"""
import torch
from basic_model import Net
from unet_model import Unet
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import os
import argparse
import matplotlib.pyplot as plt


        # Use the input transform to convert images to grayscale
input_transform = T.Compose([
                                          T.Resize(size=(256,256)),
                                          T.Grayscale(),
                                          T.ToTensor(),
                                          T.Normalize((0.5,), (0.5,))
                                          ])
        # Use this on target images(colorful ones)
target_transform = T.Compose([
                                           T.Resize(size=(256,256)),
                                           T.ToTensor(),
                                           T.Normalize((0.5,), (0.5,))])
    
    
    
    
if __name__ == '__main__':
    #model=basic_model.Net()
    parser = argparse.ArgumentParser(prog='predict')
    parser.add_argument('--image', default='data/val/0.jpg', type=str, help='the path of test image')
    parser.add_argument('--model', default='basic',type=str, help='which model')
    args = parser.parse_args()
    image_path=args.image
    model_path=args.model
    #image_path='data/val/0.jpg'
    # load the test image and transfer it
    test_img=Image.open(image_path)
    img_shape=test_img.size
    test_img=input_transform(test_img)
    # load the pretrained model
    #model=Net()
    model=Unet()
    # load the weights
    model_path=os.path.join(model_path,'trained.pth')
    pretrained=torch.load(model_path)
    model.load_state_dict(pretrained)
    model.eval()
    pred_test=model(test_img[None,...]) # colorize the image
    pred_test = F.interpolate(pred_test, size=(img_shape[1],img_shape[0]), mode='bilinear')
    pred_test=pred_test[0].cpu().detach().numpy()
    pred_test=pred_test.transpose(1,2,0) # transfer the order to RGB format
    pred_test=pred_test*0.5+0.5# denormalize the imag
    int_out=pred_test.astype('uint8')
    plt.imsave('test.png',int_out)
    plt.imshow(pred_test)
    plt.axis('off')



