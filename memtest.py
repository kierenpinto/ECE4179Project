import torch
import os
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as Transforms
import matplotlib.pyplot as plt
import u_net

# Check out how much memory is being used by a single image forward prop through network.
input_folder = '../Image_crops/'
image = cv2.imread(input_folder+'slide001_core004_crop0024.png')
plt.imshow(image)
model = u_net.unet().float()
transform = Transforms.ToTensor()
image = transform(image/255).float()

model.train()
model.forward(image.unsqueeze(0))

plt.show()
