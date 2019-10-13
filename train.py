import torch.nn as nn
import torch 
import data_load
import multiprocessing
import u_net
from torch import optim
import time
import copy
import os

def accuracy(fx, y):
    ''' Calculate Accuracy of Network'''
    # preds = fx.max(1, keepdim=True)[1]
    # correct = preds.eq(y.view_as(preds).to(torch.long)).sum()
    # acc = correct.float()/preds.shape[0]
    # return acc
    return 0

def save_model(Save_Path,model,epoch,optimizer,train_acc):
    torch.save({
        'epoch':                 epoch,
        'model_state_dict':      model.state_dict(),
        'optimizer_state_dict':  optimizer.state_dict(), 
        'train_acc':             train_acc,
        # 'valid_acc':             valid_acc,
        }, Save_Path)

def train(Save_Path,model,optimizer,device,loss_fn,dataloader,best_acc = 0, start_epoch=0,n_epochs=20):
    ''' Train the Network '''
    train_acc = 1
    # Iterate through each epoch
    for epoch in range(start_epoch,n_epochs):
        print("Running Epoch No: {}".format(epoch))
        running_loss_train = 0.0
        start_time = time.time()
        model.train() # put into training mode
        for i, (image, target) in enumerate(dataloader):   #For each training batch...
            print("running batch no: {}".format(i))
            image = image.to(device) #Move images and targets to device
            target = target.to(device)
            outputs = model(image) #Forward pass through model
            # print(outputs.permute(0,2,3,1).view(-1,2).shape)
            new_outputs = outputs.permute(0,2,3,1).view(-1,6)
            new_targets = target.view(-1).long()
            # print(target.view(-1).shape)
            loss = loss_fn(new_outputs,new_targets)#Compute cross entropy loss
            running_loss_train += loss.item()
            optimizer.zero_grad()#Gradients are accumulated, so they should be zeroed before calling backwards
            loss.backward()#Backward pass through model and update the model weights
            optimizer.step()  
        running_loss_train /= len(dataloader)
        # train_acc = calculate_accuracy(torch.cat(labels_predict),torch.cat(labels))
        end_time = time.time()
        # Store Best Model
        if train_acc > best_acc:
            best_acc = train_acc
            save_model(Save_Path,model,epoch,optimizer,train_acc)
        print('[Epoch {0:02d}] Train Loss: {1:.4f}, Train Acc: {2:.4f}, Time: {3:.4f}s'.format(
            epoch, running_loss_train,train_acc,end_time - start_time))

def main():
    #Set GPU device if available
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    # Initialise Model
    model = u_net.unet().to(device)
    Start_From_Checkpoint = True
    #Initialise Dataset
    input_folder = '../Image_crops/'
    target_folder = '../Map_crops/'
    batch_size = 5
    n_workers = multiprocessing.cpu_count()
    dataset = data_load.dataset(input_folder, target_folder, model, device)

    #Initialise Dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=n_workers)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    loss_fn = nn.CrossEntropyLoss()
    save_dir = './'
    model_name = 'Unet'
    #Create Save Path from save_dir and model_name, we will save and load our checkpoint here
    Save_Path = os.path.join(save_dir, model_name + ".pt")

    #Setup defaults:
    start_epoch = 0
    best_valid_acc = 0
    #Create the save directory if it does note exist
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

        #Load Checkpoint
        if Start_From_Checkpoint:
            #Check if checkpoint exists
            if os.path.isfile(Save_Path):
                #load Checkpoint
                check_point = torch.load(Save_Path)
                #Checkpoint is saved as a python dictionary
                model.load_state_dict(check_point['model_state_dict'])
                optimizer.load_state_dict(check_point['optimizer_state_dict'])
                start_epoch = check_point['epoch']
                best_valid_acc = check_point['train_acc'] #CHANGE THIS TO VALIDATION ACCURACY
                print("Checkpoint loaded, starting from epoch:", start_epoch)
            else:
                #Raise Error if it does not exist
                raise ValueError("Checkpoint Does not exist")
        else:
            #If checkpoint does exist and Start_From_Checkpoint = False
            #Raise an error to prevent accidental overwriting
            if os.path.isfile(Save_Path):
                raise ValueError("Warning Checkpoint exists")
            else:
                print("Starting from scratch")
    train(Save_Path,model,optimizer,device,loss_fn,dataloader,best_valid_acc,start_epoch,n_epochs=2)

if __name__ == '__main__':
    main()
