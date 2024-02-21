import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        """
        My custom ResidualBlock

        [input]
        * in_channels  : input channel number
        * out_channels : output channel number
        * kernel_size  : kernel size
        * stride       : stride size

        [hint]
        * See the instruction PDF for details
        * Set the bias argument to False
        """
        
        ## Define all the layers
        # ----- TODO -----

        raise NotImplementedError

    def forward(self, x):
       
        # ----- TODO -----
        raise NotImplementedError


class MyResnet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()

        """
        My custom ResNet.

        [input]
        * in_channels  : input channel number
        * num_classes  : number of classes

        [hint]
        * See the instruction PDF for details
        * Set the bias argument to False
        """
        
        ## Define all the layers
        # ----- TODO -----
        raise NotImplementedError


    def forward(self, x, return_embed=False):
        """
        Forward path.

        [input]
        * x             : input data
        * return_embed  : whether return the feature map of the last conv layer or not

        [output]
        * output        : output data
        * embedding     : the feature map after the last conv layer (optional)
        
        [hint]
        * See the instruction PDF for network details
        * You want to set return_embed to True if you are dealing with CAM
        """

        # ----- TODO -----
        raise NotImplementedError


def init_weights_kaiming(m):

    """
    Kaming initialization.

    [input]
    * m : torch.nn.Module

    [hint]
    * Refer to the course slides/recitations for more details
    * Initialize the bias term in linear layer by a small constant, e.g., 0.01
    """

    if isinstance(m, nn.Conv2d):
        # ----- TODO -----
        raise NotImplementedError

    elif isinstance(m, nn.Linear):
        # ----- TODO -----
        raise NotImplementedError


if __name__ == "__main__":

    # set model
    net = MyResnet(in_channels=3, num_classes=10)
    net.apply(init_weights_kaiming)
    
    # sanity check
    input = torch.randn((64, 3, 32, 32), requires_grad=True)
    output = net(input)
    print(output.shape)