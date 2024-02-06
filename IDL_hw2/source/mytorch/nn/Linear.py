import numpy as np
from mytorch.nn.initialization import Xavier, He

'''
inspiration: https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear'''
class Linear:
    
    def __init__(self, dim_in, dim_out, initialization=None, gain=1.0, debug = False):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.gain = gain
        
        if initialization == 'xavier':
            xavier_init = Xavier()
            self.W = xavier_init.initialize(self.dim_out,self.dim_in,  self.gain)
        elif initialization == 'he':
            he_init = He()
            self.W = he_init.initialize(self.dim_out,self.dim_in,  self.gain)
        else:
            self.W = np.random.randn(dim_out,dim_in)
        
        self.b = np.zeros((self.dim_out,1))
        self.dLdW = None
        self.dLdb = None
        self.debug = debug
    
    def forward(self, A):
        # A shape (batch, dim_in)
        self.A = A        
        self.batch = A.shape[0]  # batch size
        # (batch, dim_in) @ (dim_in, dim_out) + (batch, 1) @ (1, dim_out)
        Z = A @ self.W.T +  np.ones((self.batch,1))@ self.b.T
        return Z
    
    def backward(self, dLdZ):
        # A loss wrt to output
        """
        - dLdA = dLdZ dZdI = dLdZ @ W.T
        - dLdW = dLdZ dZdW = dLdZ @ A.T
        - dLdb = dLdZ dZdb = dLdZ @ 1
        """
        dLdA = dLdZ @ self.W
        self.dLdW = dLdZ.T @ self.A
        self.dLdb = np.sum(dLdZ.T, axis=1, keepdims=True)
        # self.dLdb = dLdZ.T @ np.ones((self.batch,1))
        # had to transpose this, then was getting error 
        if self.debug is True:
            self.dLdA = dLdA
        
        return dLdA
    
    # @property #(looked it on gpt)
    # def parameters(self):
    #     return [{'params': self.W, 'grad': self.dLdW}, {'params': self.b, 'grad': self.dLdb}]
    
    # @property
    # def parameters(self):
    #     params_list = [{'params': self.W, 'grad': None}, {'params': self.b, 'grad': None}]
    #     if hasattr(self, 'dLdW') and hasattr(self, 'dLdb'):
    #         params_list[0]['grad'] = self.dLdW
    #         params_list[1]['grad'] = self.dLdb
    #     return params_list