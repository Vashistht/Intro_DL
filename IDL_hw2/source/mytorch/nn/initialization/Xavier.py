'''
Implement Xavier initialization here
Source: https://pytorch.org/docs/stable/_modules/torch/nn/init.html#xavier_uniform_

- also HW 6, question 3 to Intro to ML (18661)
'''

import numpy as np

class Xaiver:
    def __init__(self):
        pass
    
    def initialize(self, dim_in, dim_out, gain=1.0):
        '''
        Initialize the input tensor with Xavier initialization
        '''
        # Xavier initialization
        # get dim in, dim out 
        # dim_in, dim_out = tensor.size(0), tensor.size(1)
        bound = gain* np.sqrt(6/(dim_in + dim_out))
        W = np.random.uniform(-bound, bound, size = (dim_in, dim_out) )
        return W
    
    