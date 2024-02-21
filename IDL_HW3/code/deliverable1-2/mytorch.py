import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

# added: https://github.com/jariasf/CS231n/blob/master/assignment2/cs231n/fast_layers.py

def MyFConv2D(input, weight, bias=None, stride=1, padding=0):
    
    """
    My custom Convolution 2D calculation.

    [input]
    * input    : (batch_size, in_channels, input_height, input_width)
    * weight   : (you have to derive the shape :-)
    * bias     : bias term
    * stride   : stride size
    * padding  : padding size

    [output]
    * output   : (batch_size, out_channels, output_height, output_width)
    """

    assert len(input.shape) == len(weight.shape) , "weight shape incorrect"
    assert len(input.shape) == 4, "input shape incorrect"
    
    k = weight.shape[-1]
    
    # batch_size, in_channels, input_height, input_width = input.shape
    N, C_in, H_in, W_in = input.shape
    # weight shape  (C_out, C_in, kH, kW)
    C_out, C_in, kH, kW = weight.shape 
    b, s, p = bias, stride, padding

    x_pad =  F.pad(input,(p,p,p,p),mode='constant',value=0) # add 2p to last two dims of input
    assert x_pad.shape == (N, C_in, H_in + 2*p, W_in + 2*p) , "padding incorrect"

    # k = 0 # kernel size
    H_out = int( (H_in + 2*p - kH) / s) + 1
    W_out = int( (W_in + 2*p - kW) / s) + 1
    
    ## Derive the output size
    ## Create the output tensor and initialize it with 0
    output = torch.zeros(N, C_out, H_out, W_out)
    
    ## Convolution process
    for b in range(N):
        for c_o in range(C_out):
            for h in range(H_out):
                for w in range(W_out):
                    # get the window
                    window = x_pad[b, :, h*s:h*s+kH, w*s:w*s+kW]
                    # apply the kernel: element-wise multiplication
                    # (b,c_in, kH, KW) * (c_o, c_i, kH, kW) -> (b, c_o, kH, kW)
                    output[b, c_o, h, w] = torch.sum(window * weight[c_o,:,:,:])
                    if bias is not None:
                        output[b, c_o, h, w] += bias[c_o]
    return output
    ## Feel free to use for loop 


class MyConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):

        """
        My custom Convolution 2D layer.

        [input]
        * in_channels  : input channel number
        * out_channels : output channel number
        * kernel_size  : kernel size
        * stride       : stride size
        * padding      : padding size
        * bias         : taking into account the bias term or not (bool)

        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        ## Create the torch.nn.Parameter for the weights and bias (if bias=True)
        ## Be careful about the size
        # ----- TODO -----
        self.W = None 
        self.b = None

        # out = ( (2*p + H - k) / s ) + 1
        
        raise NotImplementedError
            
    
    def __call__(self, x):
        
        return self.forward(x)


    def forward(self, x):
        
        """
        [input]
        x (torch.tensor)      : (batch_size, in_channels, input_height, input_width)

        [output]
        output (torch.tensor) : (batch_size, out_channels, output_height, output_width)
        """

        # call MyFConv2D here
        # ----- TODO -----
        
        raise NotImplementedError

    
class MyMaxPool2D(nn.Module):

    def __init__(self, kernel_size, stride=None):
        
        """
        My custom MaxPooling 2D layer.
        [input]
        * kernel_size  : kernel size
        * stride       : stride size (default: None)
        """
        super().__init__()
        self.kernel_size = kernel_size

        ## Take care of the stride
        ## Hint: what should be the default stride_size if it is not given? 
        ## Think about the relationship with kernel_size
        # ----- TODO -----
        self.stride = None

        raise NotImplementedError


    def __call__(self, x):
        
        return self.forward(x)
    
    def forward(self, x):
        
        """
        [input]
        x (torch.tensor)      : (batch_size, in_channels, input_height, input_width)

        [output]
        output (torch.tensor) : (batch_size, out_channels, output_height, output_width)

        [hint]
        * out_channel == in_channel
        """
        
        ## check the dimensions
        self.batch_size = x.shape[0]
        self.channel = x.shape[1]
        self.input_height = x.shape[2]
        self.input_width = x.shape[3]
        
        ## Derive the output size
        # ----- TODO -----
        self.output_height   = None
        self.output_width    = None
        self.output_channels = None
        self.x_pool_out      = None

        ## Maxpooling process
        ## Feel free to use for loop
        # ----- TODO -----

        raise NotImplementedError


if __name__ == "__main__":

    ## Test your implementation of MyFConv2D.
    # ----- TODO -----
    pass
