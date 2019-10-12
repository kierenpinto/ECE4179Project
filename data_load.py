import torch
import os
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as Transforms

class dataset(Dataset):
    def __init__(self, input_folder, target_folder, model):
        super(dataset,self).__init__()
        self.input_folder = input_folder
        self.target_folder = target_folder

        self.inputs = [x for x in os.listdir(input_folder)]
        self.targets = [x for x in os.listdir(target_folder)]
        h, w, c = cv2.imread("{}{}".format(input_folder,self.inputs[0])).shape
        self.model = model

        self.resize_shape = model(torch.zeros(1, c, h, w)).shape

        self.transform = Transforms.ToTensor()

    def __getitem__(self, index):
        image = cv2.imread(self.input_folder+self.inputs[index])
        image = self.transform(image)
        target = cv2.imread(self.target_folder+self.targets[index])
        target = cv2.resize(target, (self.resize_shape[2:]))
        target = self.transform(target)[0]
        return [image, target]

    def __len__(self):
        return len(self.inputs)