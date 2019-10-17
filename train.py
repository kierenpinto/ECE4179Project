import torch.nn as nn
import torch 
from data_load import dataset
import multiprocessing
import u_net
from torch import optim
import time
import copy
import os
import numpy as np

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1] #Argmax
    correct = preds.eq(y.view_as(preds).to(torch.long)).sum()
#     print("prediction",preds)
#     print("y",y)
    return correct

def test(model, loader,loss_fn,device):
    '''Calculates accuracy on a dataset'''
    model.eval()
    num_correct = []
    target_shape = []
    running_loss = 0.0
    with torch.no_grad():
        for i, (image, target) in enumerate(loader):
            print("Forward batch {}".format(i))
            image = image.to(device)
            target = target.to(device).view(-1) # [5,324,324]->[524880]
            target_predict = model.forward(image).permute(0,2,3,1).reshape(-1,4) #[5,4,324,324]->[5,324,324,4]->[524880,4]
            loss = loss_fn(target_predict,target.to(torch.long))
            running_loss += loss.item()
            num_correct.append(calculate_accuracy(target_predict, target))
            target_shape.append(target.shape[0])
            del loss
            del target_predict
            del target
            torch.cuda.empty_cache()
    running_loss /= len(loader)
    accuracy = (torch.tensor(num_correct).sum().float()/(sum(target_shape))).item()
    # print(target_shape)
    # print(num_correct)
    return running_loss, accuracy

def save_model(Save_Path,model,epoch,optimizer,train_loss, train_acc, valid_loss ,valid_acc, best_acc):
    torch.save({
        'epoch':                 epoch,
        'model_state_dict':      model.state_dict(),
        'optimizer_state_dict':  optimizer.state_dict(),
        'train_loss':            train_loss,
        'train_acc':             train_acc,
        'valid_loss':            valid_loss,
        'valid_acc':             valid_acc,
        'best_acc':              best_acc,
        }, Save_Path)

def train(save_dir,model_name,model,optimizer,device,loss_fn,dataloader,valloader, best_acc = 0, start_epoch=0,n_epochs=20, train_loss = [],train_acc = [], val_loss = [], val_acc = []):
    ''' Train the Network '''
    # Iterate through each epoch
    for epoch in range(start_epoch,n_epochs):
        print("Running Epoch No: {}".format(epoch))
        running_loss_train = 0.0
        start_time = time.time()
        num_correct = []
        target_shape = []
        model.train() # put into training mode
        for i, (image, target) in enumerate(dataloader):   #For each training batch...
            print("running batch no: {}".format(i))
            image = image.to(device) #Move images and targets to device
            target = target.to(device)
            outputs = model(image) #Forward pass through model
            new_outputs = outputs.permute(0,2,3,1).reshape(-1,4)
            new_targets = target.view(-1).long()
            loss = loss_fn(new_outputs,new_targets)#Compute cross entropy loss
            running_loss_train += loss.item()
            num_correct.append(calculate_accuracy(new_outputs, new_targets))
            target_shape.append(new_targets.shape[0])
            optimizer.zero_grad()#Gradients are accumulated, so they should be zeroed before calling backwards
            loss.backward()#Backward pass through model and update the model weights
            optimizer.step()  
        running_loss_train /= len(dataloader)
        train_loss.append(running_loss_train) # Training Loss
        # print("sum num correct",torch.tensor(num_correct).sum().float())
        # print("target_shape", target_shape)
        # print("target shape sum", sum(target_shape))
        train_acc.append((torch.tensor(num_correct).sum().float()/(sum(target_shape))).item()) # Append Training Accuracy
        vloss, vacc = test(model,valloader,loss_fn,device) # Run test
        val_loss.append(vloss) # Append validation loss
        val_acc.append(vacc) # Append validation accuracy
        end_time = time.time()
        # Store Best Model
        if val_acc[-1] > best_acc:
            best_acc = val_acc[-1] # Save on best epoch
            Save_Path = os.path.join(save_dir, model_name + "_" + str(epoch) + "_best.pt")
            save_model(Save_Path,model,epoch,optimizer,train_loss,train_acc,val_loss,val_acc,best_acc)
        if (not epoch%5 ): # Save every 5 epochs
            Save_Path = os.path.join(save_dir, model_name + "_" + str(epoch) + ".pt")
            save_model(Save_Path,model,epoch,optimizer,train_loss,train_acc,val_loss,val_acc,best_acc)

        print('[Epoch {0:02d}] Train Loss: {1:.4f}, Train Acc: {2:.4f}, Val Loss: {3:.4f}, Val Acc: {4:.4f}, Time: {5:.4f}s'.format(
            epoch, running_loss_train,train_acc[-1], val_loss[-1], val_acc[-1],end_time - start_time))
    return train_loss, train_acc, val_loss, val_acc

def main():
    #Set GPU device if available
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    # Initialise Model
    model = u_net.unet().to(device)
    def initweights(layer):
  
    if type(layer)==nn.Conv2d:
        kernsize=layer.kernel_size
        cin=layer.in_channels
        N=cin*kernsize[0]*kernsize[0]
        standard=math.sqrt(2/N)
        torch.nn.init.normal_(layer.weight,std=standard)
    model.apply(initweights)
    
    Start_From_Checkpoint = False
    #Initialise Dataset
    input_folder = '/mnt/lustre/projects/ds19/eng121/Image_crops/'
    target_folder = '/mnt/lustre/projects/ds19/eng121/Map_crops/'
    batch_size = 5
    n_workers = multiprocessing.cpu_count()
    trainset = dataset(input_folder, target_folder, model,device,False)
    valset = dataset(input_folder,target_folder,model,device,True)
    #Initialise Dataloader
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=n_workers)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=True, num_workers=n_workers)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    loss_fn = nn.CrossEntropyLoss()
    save_dir = './'
    model_name = 'Unet'
    #Create Save Path from save_dir and model_name, we will save and load our checkpoint here
    # Save_Path = os.path.join(save_dir, model_name + ".pt")
    Load_Path = os.path.join(save_dir, model_name + ".pt")

    #Setup defaults:
    start_epoch = 0
    best_valid_acc = 0
    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    #Create the save directory if it does note exist
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    #Load Checkpoint
    if Start_From_Checkpoint:
        #Check if checkpoint exists
        if os.path.isfile(Load_Path):
            #load Checkpoint
            check_point = torch.load(Load_Path)
            #Checkpoint is saved as a python dictionary
            model.load_state_dict(check_point['model_state_dict'])
            optimizer.load_state_dict(check_point['optimizer_state_dict'])
            start_epoch = check_point['epoch']
            best_valid_acc = check_point['best_acc']
            train_loss = check_point['train_loss']
            train_acc = check_point['train_acc']
            valid_loss = check_point['valid_loss']
            valid_acc = check_point['valid_acc']
            
            print("Checkpoint loaded, starting from epoch:", start_epoch)
        else:
            #Raise Error if it does not exist
            raise ValueError("Checkpoint Does not exist")
    else:
        #If checkpoint does exist and Start_From_Checkpoint = False
        #Raise an error to prevent accidental overwriting
        if os.path.isfile(Load_Path):
            raise ValueError("Warning Checkpoint exists")
        else:
            print("Starting from scratch")
    train(save_dir,model_name,model,optimizer,device,loss_fn,dataloader,valloader,best_valid_acc,start_epoch,n_epochs=20,train_loss = train_loss,train_acc=train_acc, val_loss = valid_loss, val_acc = valid_acc)

if __name__ == '__main__':
    main()
