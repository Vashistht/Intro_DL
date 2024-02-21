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
        
        ## Define all the layers
        ## Use MyConv2D, MyMaxPool2D for the network
        # ----- TODO -----

        raise NotImplementedError


    def forward(self, x):
        
        # ----- TODO -----

        raise NotImplementedError


if __name__ == "__main__":

    # set param
    setup_seed(18786)
    batch_size = 128
    num_epoch = 1
    lr = 1e-4

    ## Load dataset

    # ----- TODO -----
    trainset = None
    trainloader = None
    valset = None
    valloader = None

    print(f"LOAD DATASET: TRAIN {len(trainset)} | TEST: {len(valset)}")

    ## Load my neural network
    # ----- TODO -----
    

    ## Define the criterion and optimizer
    # ----- TODO -----

    ## Training and evaluation
    ## Feel free to record the loss and accuracy numbers
    ## Hint: you could separate the training and evaluation
    ## process into 2 different functions for each epoch
    for epoch in range(num_epoch): 
        # ----- TODO -----
        pass

    ## Plot the loss and accuracy curves
    # ----- TODO -----



