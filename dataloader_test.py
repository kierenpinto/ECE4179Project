import data_load
import torch
import u_net
input_folder = '../Image_crops/'
target_folder = '../Map_crops/'

model = u_net.unet()
dataset = data_load.dataset(input_folder, target_folder, model)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=10,shuffle=True, num_workers=1)
# print(next(enumerate(dataloader))[1][1].dtype)
print(next(enumerate(dataloader))[1][1])
# print(model.encoder0.layer.end=)