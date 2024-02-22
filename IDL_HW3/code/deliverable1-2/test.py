import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from mytorch import MyFConv2D, MyConv2D, MyMaxPool2D


print("Testing MyConv2D")
# Create inputs for validation
batch_size, in_channels, H, W = 4,5,8,8 # Example dimensions
out_channels = 3
kernel_size = 3
stride = 1
padding = 2



# Input and weights
input_tensor = torch.rand(batch_size, in_channels, H, W)
weights = torch.rand(out_channels, in_channels, kernel_size, kernel_size)
bias = torch.zeros(out_channels)

myConvd2D = MyConv2D(in_channels, out_channels, kernel_size, stride, padding)
conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
conv2d.weight.data = myConvd2D.W.data.clone()
conv2d.bias.data = myConvd2D.b.data.clone()

# Apply PyTorch Conv2D
with torch.no_grad():  # Ensure Conv2D weight and bias are not updated
    torch_conv_output = conv2d(input_tensor)


my_conv_output = myConvd2D(input_tensor)

# Compare outputs
print("Output Difference:", torch.sum(torch_conv_output - my_conv_output))
print('__________________________________________________________')

print('Testing MyFConv2D')
# Sample input tensor
input_tensor = torch.randn(2, 4, 8, 8)

# Custom max pool layer
my_maxpool = MyMaxPool2D(kernel_size=2, stride=2)

# PyTorch max pool layer
torch_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

# Apply pooling
custom_pool_output = my_maxpool(input_tensor)
torch_pool_output = torch_maxpool(input_tensor)

# Compare outputs
print("Difference between custom and torch max pooling outputs:", torch.norm(custom_pool_output - torch_pool_output).item())

print('__________________________________________________________')

