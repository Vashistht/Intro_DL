import os
import time
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from model import MyResnet, init_weights_kaiming


def setup_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cam(net, inputs, labels, idx):

    """
    Calculate the CAM.

    [input]
    * net     : network
    * inputs  : input data
    * labels  : label data
    * idx     : the index of the chosen image in a minibatch, range: [0, batch_size-1]

    [output]
    * cam_img : CAM result
    * img     : raw image

    [hint]
    * Inputs and labels are in a minibatch form
    * You can choose one images from them for CAM by idx.
    """

    net.eval()
    DEVICE = torch.device("mps:0" if torch.cuda.is_available() else "cpu") # was cuda:0
    with torch.no_grad():
        
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs, feat_map = net(inputs, return_embed=True)
        # feat_map: torch.Size([128, 512, 4, 4]), outputs: torch.Size([128, 10])
        softmax_output = F.softmax(outputs, dim=-1)
        
        ## Find the class with highest probability
        pred_class  = torch.argmax(softmax_output[idx,:], dim=0)
        
        ## Obtain the weight related to that class
        weight = net.fc.weight # weight: torch.Size([10, 512])
        weight = weight[pred_class,:] # weight: torch.Size([512])
        # weight = weight.reshape(1,-1) # weight: torch.Size([1, 512])
        # weight = weight.unsqueeze(-1).unsqueeze(-1)
        weight = weight.reshape(1, -1, 1, 1) # torch.Size([1, 512])
        # get feature map for the idx 
        feat_map = feat_map[idx, :, :, :] # torch.Size([512, 4, 4])
        feat_map = feat_map.unsqueeze(0) # torch.Size([1, 512, 4, 4])
        
        
        ## Calculate the CAM
        ## Hint: you can choose one of the image (idx) from the batch for the following process
        # ----- TODO -----
        cam = weight.mul(feat_map) # torch.Size([1, 512, 4, 4])
        cam = torch.sum(cam, dim=1) # sum across 512 channels (torch.Size([1, 4, 4]))


        ## Normalize CAM 
        ## Hint: Just minmax norm and rescale every value between [0-1]
        cam = (cam - torch.min(cam)) / (torch.max(cam) - torch.min(cam))
                
        ## You will want to resize the CAM result for a better visualization
        input_size = (inputs.size(2), inputs.size(3)) # raw img size
        cam_img = F.interpolate(cam.unsqueeze(0), size=input_size, mode='bilinear', align_corners=False) # returns torch.Size([1, 1, 32, 32])

        cam_img = cam_img.squeeze(0).squeeze(0)
        cam_img = cam_img.detach().cpu().numpy()
        ## Denormalize raw images
        ## Hint: reverse the transform we did before
        
        ## Change the image data type into uint8 for visualization
        img = inputs[idx].permute(1,2,0).detach().cpu().numpy()
        img = normalize_param['mean'] + normalize_param['std'] * img
        

        return cam_img, img

def train(model, trainloader, device):
    model.train()
    running_loss = correct = 0.0
    total = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += torch.sum(labels == torch.argmax(outputs, dim=-1) )
        total += labels.size(0)
    
    avg_loss = running_loss / len(trainloader)
    accuracy = correct/ total 
    return avg_loss, accuracy



def validation(model, valloader, device):
    model.eval()
    running_loss = correct = 0
    total = 0
    for i,data in enumerate(valloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        correct += torch.sum(labels == torch.argmax(outputs, dim=-1))
        total += labels.size(0)

    accuracy = correct / total
    avg_loss = running_loss / len(valloader)
    return avg_loss, accuracy


def plot_cam(cam_img, img, idx): # for plotting the cam result for a given index
    fig, ax = plt.subplots(1,3, figsize=(10,5))
    ax[0].imshow(img)
    ax[0].set_title('Raw Image')
    ax[1].imshow(cam_img, cmap='jet', alpha = .5)
    ax[1].set_title('CAM Result')
    ax[2].imshow(img)
    ax[2].imshow(cam_img, cmap='jet', alpha=0.4)
    ax[2].set_title('CAM Blended')
    plt.savefig(f'cam-result-{idx}.png')
    # plt.show()

if __name__ == "__main__":

    # set param
    setup_seed(18786)
    batch_size = 128
    num_epoch = 30
    lr = 1e-4
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # was cuda:0
    print(DEVICE)
    # device = torch.device("mps:0")
   
    ## Load dataset
    normalize_param = dict(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
        )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)), 
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), 
        transforms.Normalize(**normalize_param,inplace=True)
        ])

    val_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(**normalize_param,inplace=True)
        ])

    # getting the dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    
    valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)

    # loading the dataset
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # ran for 5 epochs on a subset of data and then ran the full dataset
    # trainloader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, range(1, 20000)), batch_size=batch_size, shuffle=True, num_workers=2)
    # valloader = torch.utils.data.DataLoader(torch.utils.data.Subset(valset, range(1, 5000)), batch_size=batch_size, shuffle=False, num_workers=2)
    
    
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print(f"LOAD DATASET: TRAIN/VAL | {len(trainset)}/{len(valset)}")
    
    ## Training and evaluation
    train_accuracy = []
    train_loss = []
    val_accuracy = []
    val_loss = []
    best_loss = 1e6
    
    net = MyResnet().to(DEVICE)
    net.apply(init_weights_kaiming)
    if os.path.exists('best_model.pth'):
        state = torch.load('best_model.pth',map_location=DEVICE)
        net.load_state_dict(state)
    net.to(DEVICE)
    
    ## Create the criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    for epoch in range(num_epoch): 
        train_loss_epoch, train_acc_epoch = train(net, trainloader, device=DEVICE)
        val_loss_epoch, val_acc_epoch = validation(net, valloader, device=DEVICE)
        
        train_accuracy.append(train_acc_epoch)
        train_loss.append(train_loss_epoch)
        
        val_accuracy.append(val_acc_epoch)
        val_loss.append(val_loss_epoch)
        if val_loss_epoch < best_loss:
            best_loss = val_loss_epoch
            torch.save(net.state_dict(), 'best_model.pth')
        
        
        print(f"Epoch {epoch+1}  | Train Loss: {train_loss_epoch:.4f} | Train Acc: {train_acc_epoch:.4f} | Val Loss: {val_loss_epoch:.4f} | Val Acc: {val_acc_epoch:.4f} | Best Val Loss: {best_loss:.4f}")
    
    print('Finished Training')
    ## Visualization
    ## Plot the loss and acc curves
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].plot(train_loss, label='Train Loss')
    ax[0].plot(val_loss, label='Val Loss')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    
    ax[1].plot(train_accuracy, label='Train Acc')
    ax[1].plot(val_accuracy, label='Val Acc')
    ax[1].set_title('Accuracy')
    ax[1].legend()
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].axhline(y=0.85, color='r', linestyle='--', label='85% cut-off')
    ax[1].legend()
    plt.savefig('train-loss-curve-del4.png')
    plt.show()

    if os.path.exists('best_model.pth'):
        state = torch.load('best_model.pth',map_location=torch.device('cpu'))
        net.load_state_dict(state)    
    net.to(DEVICE)
    # Fetch the test image for CAM
    dataiter = iter(valloader)
    inputs, labels = next(dataiter)
    inputs = inputs.to(DEVICE)
    labels = labels.to(DEVICE)
    
    for i in [0,10,25,30,50,75,90]:
        cam_img, img = cam(net, inputs, labels, idx=i) # idx could be changed
        plot_cam(cam_img, img, idx=i)