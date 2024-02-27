import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
    # batch_size, in_channels, input_height, input_width = input.shape
    N, C_in, H_in, W_in = input.shape
    # weight shape  (C_out, C_in, kH, kW)
    C_out, C_in, kH, kW = weight.shape 
    bias, s, p = bias, stride, padding

    x_pad =  F.pad(input,(p,p,p,p),mode='constant',value=0) # add 2p to last two dims of input
    assert x_pad.shape == (N, C_in, H_in + 2*p, W_in + 2*p) , "padding incorrect"

    H_out = int( (H_in + 2*p - kH) / s) + 1 # output sizes 
    W_out = int( (W_in + 2*p - kW) / s) + 1
    
    output = torch.zeros(N, C_out, H_out, W_out, device = input.device) # initalise output tensor
    
    weight = weight.to(input.device)
    bias = bias.to(input.device)
    ## Convolution
    for h in range(H_out):
        for w in range(W_out):
            
            window = x_pad[:, :, h*s:h*s+kH, w*s:w*s+kW]
            # (b,c_in, kH, KW) * (c_o, c_i, kH, kW) -> (b, c_o, _)
            # sum over c_in, kH, kW
            output[:,:, h, w] = torch.tensordot(window, weight, dims=([1,2,3],[1,2,3]))
            if bias is not None:
                output[:,:, h, w] += bias[:]
    return output



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
        # weight shape  (C_out, C_in, kH, kW)
        # N, C_in, H_in, W_in = input.shape

        self.W =  nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.b = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # out = ( (2*p + H - k) / s ) + 1            
    
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
        output = MyFConv2D(x, self.W, self.b, self.stride, self.padding).to(x.device)
        return output
    

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
        if stride is not None:
            self.stride = stride
        else:
            self.stride = kernel_size # default stride size

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
        self.output_height   = self.conv_output(self.input_height)
        self.output_width    = self.conv_output(self.input_width)
        self.output_channels = self.channel
        self.x_pool_out      = torch.zeros((self.batch_size, self.output_channels, self.output_height, self.output_width))
        
        ## Convolution
        output = torch.zeros(self.batch_size, self.output_channels, self.output_height, self.output_width, device=x.device)
        
        s, k = self.stride, self.kernel_size
        for h in range(self.output_height):
            for w in range(self.output_width):
                # B, C_o, h, w
                window = x[:, :, h*s:h*s+k, w*s:w*s+k]
                window = window.reshape(self.batch_size, self.channel, -1)
                # window = window.reshape(self.batch_size, self.channel, k*k)
                max_pool = torch.max(window, dim=(2)).values
                output[:,:, h, w] = max_pool
        return output

    def conv_output(self, dim_in): # assuming sq kernel
        dim_out = int( (dim_in - self.kernel_size) / self.stride) + 1 # output sizes 
        return dim_out

if __name__ == "__main__":
    print("Testing MyConv2D")
    # Create inputs for validation
    batch_size, in_channels, H, W = 4,5,8,8
    out_channels = 3
    kernel_size = 3
    stride = 1
    padding = 2

    input_tensor = torch.rand(batch_size, in_channels, H, W)
    weights = torch.rand(out_channels, in_channels, kernel_size, kernel_size)
    bias = torch.zeros(out_channels)

    myConvd2D = MyConv2D(in_channels, out_channels, kernel_size, stride, padding)
    conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    conv2d.weight.data = myConvd2D.W.data.clone()
    conv2d.bias.data = myConvd2D.b.data.clone()

    with torch.no_grad():
        torch_conv_output = conv2d(input_tensor)


    my_conv_output = myConvd2D(input_tensor)

    print("Output Difference:", torch.sum(torch_conv_output - my_conv_output))
    print('__________________________________________________________')

    print('Testing MyFConv2D')
    input_tensor = torch.randn(5, 5, 8, 8)

    my_maxpool = MyMaxPool2D(kernel_size=2, stride=2)

    torch_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    custom_pool_output = my_maxpool(input_tensor)
    torch_pool_output = torch_maxpool(input_tensor)

    print("Difference between custom and torch max pooling outputs:", torch.norm(custom_pool_output - torch_pool_output).item())

    print('__________________________________________________________')
