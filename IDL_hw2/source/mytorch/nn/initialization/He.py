'''
Implement Xavier initialization here
Source: https://pytorch.org/docs/stable/nn.init.html
(torch.nn.init.kaiming_uniform_)
'''

import numpy as np

class He:
    def __init__(self):
        pass
    
    def initialize(self, dim_in, dim_out, gain=1.0, mode='fan_in'):
        '''
        Initialize the input tensor with Xavier initialization
        '''
        # Xavier initialization
        # get dim in, dim out 
        # dim_in, dim_out = tensor.size(0), tensor.size(1)
        # mode_dim = (dim_in if mode == 'fan_in' else dim_out)
        # bound = gain* np.sqrt(3/ mode_dim)
        # W = np.random.uniform(-bound, bound, size = (dim_out,dim_in) )
        
        # dim_in, dim_out = tensor.size(0), tensor.size(1)
        mode_dim = (dim_in if mode == 'fan_in' else dim_out)
        bound = gain* np.sqrt(3/ mode_dim)
        W = np.random.uniform(-bound, bound, size = (dim_in, dim_out) )
        
        
        return W
