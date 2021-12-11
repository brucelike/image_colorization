from typing import Tuple
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import numpy as np
from torchvision.transforms.functional import resize
import os
import PIL

class ColorizeData(Dataset):
    def __init__(self, image_paths):
        # Initialize dataset, you may use a second dataset for validation if required
        # Use the input transform to convert images to grayscale
        self.input_transform = T.Compose([
                                          T.Resize(size=(256,256)),

                                          T.Grayscale(),
                                          T.ToTensor(),
                                          T.Normalize((0.5,), (0.5,))
                                          ])
        # Use this on target images(colorful ones)
        self.target_transform = T.Compose([
                                           T.Resize(size=(256,256)),
                                           T.ToTensor(),
                                           T.Normalize((0.5,), (0.5,))])
        self.path=image_paths
        self.image_files = os.listdir(image_paths)
    def __len__(self) -> int:
        # return Length of dataset
        return len(self.image_files)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return the input tensor and output tensor for training
        image_name = os.path.join(self.path, self.image_files[index])  
        image = PIL.Image.open(image_name)
        #if self.input_transform is not None:
        img_gray = self.input_transform(image)
        #if self.target_transform is not None:
        img_rgb = self.target_transform(image)
        return img_gray, img_rgb
