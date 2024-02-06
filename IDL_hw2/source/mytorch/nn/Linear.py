import numpy as np
from mytorch.nn.initialization import Xavier, He


class Linear:
    
    def __init__(self, dim_in, dim_out, initialization=None, gain=1.0):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.gain = gain
        
        if initialization == 'xavier':
            xavier_init = Xavier()
            self.W = xavier_init.initialize(self.dim_in, self.dim_out, self.gain)
        elif initialization == 'he':
            he_init = He()
            self.W = he_init.initialize(self.dim_in, self.dim_out, self.gain)
        else:
            self.W = np.random.randn(dim_in, dim_out)
        
        self.b = np.zeros((1, self.dim_out))
        self.dLdW = 0.0
        self.dLdb = 0.0
        
    def forward(self, input):
        # input shape (batch, dim_in)
        self.input = input        
        self.batch = input.shape[0]  # batch size
        # (batch, dim_in) @ (dim_in, dim_out) + (batch, 1) @ (1, dim_out)
        Z = input @ self.W +  np.ones((self.batch,1))@ self.b 
        return Z
    
    def backward(self, dLdZ):
        # input loss wrt to output
        """
        - dLdI = dLdZ dZdI = dLdZ @ W.T
        - dLdW = dLdZ dZdW = dLdZ @ I.T
        - dLdb = dLdZ dZdb = dLdZ @ 1
        """
        dLdI = dLdZ @ self.W.T
        self.dLdW = self.input.T @ dLdZ
        self.dLdb = np.sum(dLdZ, axis=0, keepdims=True)
        # self.dLdb = dLdZ.T @ np.ones((self.batch,1))
        # had to transpose this, then was getting error 
        return dLdI
    
    # @property #(looked it on gpt)
    # def parameters(self):
    #     return [{'params': self.W, 'grad': self.dLdW}, {'params': self.b, 'grad': self.dLdb}]
    
    @property
    def parameters(self):
        params_list = [{'params': self.W, 'grad': None}, {'params': self.b, 'grad': None}]
        if hasattr(self, 'dLdW') and hasattr(self, 'dLdb'):
            params_list[0]['grad'] = self.dLdW
            params_list[1]['grad'] = self.dLdb
        return params_list