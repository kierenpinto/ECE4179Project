import torch.nn as nn
import torch 
import data_load
import multiprocessing
import u_net
from torch import optim
import time
import copy

def accuracy(fx, y):
    ''' Calculate Accuracy of Network'''
    # preds = fx.max(1, keepdim=True)[1]
    # correct = preds.eq(y.view_as(preds).to(torch.long)).sum()
    # acc = correct.float()/preds.shape[0]
    # return acc
    return 0

def train(model,optimizer,device,loss_fn,dataloader,n_epochs=20):
    ''' Train the Network '''
    #Create empty lists to store losses
    Train_loss = []
    Train_acc = []
    best_acc = 0
    train_acc = 1
    # Iterate through each epoch
    for epoch in range(n_epochs):
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
        Train_loss.append(running_loss_train)
        # train_acc = calculate_accuracy(torch.cat(labels_predict),torch.cat(labels))
        # Train_acc.append(train_acc)

        # Store Best Model
        if train_acc > best_acc:
            best_acc = train_acc
            model_dict = model.state_dict()
            Best_model = copy.deepcopy(model_dict)
        end_time = time.time()
        torch.save(model.state_dict(),'model')
        # print('[Epoch {0:02d}] Train Loss: {1:.4f}, Train Acc: {2:.4f}, Time: {3:.4f}s'.format(
        #     epoch, running_loss_train,train_acc,end_time - start_time))
    return Train_loss, Train_acc, Best_model

def main():
    #Set GPU device if available
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    # Initialise Model
    model = u_net.unet().to(device)

    #Initialise Dataset
    input_folder = '../Image_crops/'
    target_folder = '../Map_crops/'
    batch_size = 1
    n_workers = multiprocessing.cpu_count()
    dataset = data_load.dataset(input_folder, target_folder, model)

    #Initialise Dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=n_workers)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    loss_fn = nn.CrossEntropyLoss()
    train(model,optimizer,device,loss_fn,dataloader)

if __name__ == '__main__':
    main()