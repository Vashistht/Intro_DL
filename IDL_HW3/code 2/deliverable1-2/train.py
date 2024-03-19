import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from mytorch import MyConv2D, MyMaxPool2D
import time 


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Net(nn.Module):
    def __init__(self):

        """
        My custom network
        [hint]
        * See the instruction PDF for details
        * Only allow to use MyConv2D and MyMaxPool2D
        * Set the bias argument to True
        """
        super().__init__()
        ## Use MyConv2D, MyMaxPool2D for the network
        # Conv2D* in_channels, out_channels, kernel_size, stride, padding, bias=True
        # MaxPool2D* kernel_size, stride
        
        self.conv1 = MyConv2D(1, 3, 3,1,1, True)
        self.maxpool = MyMaxPool2D(2,2)
        self.conv2 = MyConv2D(3, 6, 3,1,1, True)
        self.activation = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(6*7*7, 128)
        self.linear2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # layer1 
        x = self.conv1(x)
        x = self.activation(x)
        x = self.maxpool(x)
        # layer 2
        x = self.conv2(x)
        x = self.activation(x)
        x = self.maxpool(x)
        # flatten, Lin, Act, Lin sequence
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

# def accuracy(labels, predictions):
#     correct = torch.sum(labels == predictions)
#     accuracy = correct / len(labels)
#     return accuracy

def train(model, trainloader):
    model.train()
    running_loss = correct = 0.0
    total = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        # inputs, labels = data
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



def validation(model, valloader):
    model.eval()
    running_loss = correct = 0
    total = 0
    for i,data in enumerate(valloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        # inputs, labels = data
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        correct += torch.sum(labels == torch.argmax(outputs, dim=-1))
        total += labels.size(0)

    accuracy = correct / total
    avg_loss = running_loss / len(valloader)
    return avg_loss, accuracy


if __name__ == "__main__":

    # set param
    setup_seed(18786)
    batch_size = 128
    num_epoch = 7
    lr = 1e-4

    # if torch.backends.mps.is_available():
    #     print("Using MPS")
    #     device = torch.device("mps")
    # else:
    #     print("MPS not available, using CPU")
    device = torch.device("cpu")
    
    ## Load dataset

    # ----- TODO -----
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    valset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)


    print(f"LOAD DATASET: TRAIN {len(trainset)} | TEST: {len(valset)}")
    start = time.time()
    
    ## Load my neural network
    model = Net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training and evaluation
    ## Feel free to record the loss and accuracy numbers
    train_accuracy = []
    train_loss = []
    val_accuracy = []
    val_loss = []
    ## Hint: you could separate the training and evaluation
    ## process into 2 different functions for each epoch
    for epoch in range(num_epoch): 
        train_loss_epoch, train_acc_epoch = train(model, trainloader)
        val_loss_epoch, val_acc_epoch = validation(model, valloader)
        
        train_accuracy.append(train_acc_epoch)
        train_loss.append(train_loss_epoch)
        
        val_accuracy.append(val_acc_epoch)
        val_loss.append(val_loss_epoch)
        
        print(f"Epoch {epoch+1}  | Train Loss: {train_loss_epoch:.4f} | Train Acc: {train_acc_epoch:.4f} | Val Loss: {val_loss_epoch:.4f} | Val Acc: {val_acc_epoch:.4f}")
    end = time.time()
    print(f"Training time: {end-start:.4f}s")
    
    
     ## Plot the loss and accuracy curves
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
    ax[1].axhline(y=0.9, color='r', linestyle='--', label='90% cut-off')
    ax[1].legend()
    plt.savefig('train-loss-curve-del1.png')
    plt.show()
 