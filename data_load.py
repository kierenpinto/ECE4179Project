import torch
import os
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as Transforms
import numpy as np

class dataset(Dataset):
    def __init__(self, input_folder, target_folder, model,device,validation):
        super(dataset,self).__init__()
        self.input_folder = input_folder
        self.target_folder = target_folder
        if validation:
            ''' Uses validation set'''
            self.inputs = os.listdir(input_folder)[5000:]
            self.targets = os.listdir(target_folder)[5000:]
        else:
            ''' uses training set '''
            self.inputs = os.listdir(input_folder)[0:2000]
            self.targets = os.listdir(target_folder)[0:2000]
        h, w, c = cv2.imread("{}{}".format(input_folder,self.inputs[0])).shape
        self.model = model

        self.resize_shape = model(torch.zeros(1, c, h, w).to(device)).shape

        self.transform = Transforms.ToTensor()

    def __getitem__(self, index):
        image = cv2.imread(self.input_folder+self.inputs[index])
        image = self.transform(image)
        target = cv2.imread(self.target_folder+self.targets[index])
        target = cv2.resize(target, (self.resize_shape[2:]))
        target = np.maximum(target.astype(int)-2,0).transpose(2,0,1)[0]# Convert target from 0,3,4,5 to 0,1,2,3 therefore allowing four classes
        return [image, target]

    def __len__(self):
        return len(self.inputs)