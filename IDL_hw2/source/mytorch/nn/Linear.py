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
        elif initialization =='he':
            he_init = He()
            self.W = he_init.initialize(self.dim_in, self.dim_out, self.gain)
        else:
            self.W = np.random.randn(dim_in, dim_out)
        
        self.b = np.zeros((1, dim_out))

    def forward(self, input):
        # input shape  (batch,dim_in)
        self.input = input        
        self.batch = input.shape[0] # batch size
        
        # (batch*dim_in) @ (dim_in*dim_out) + (batch*1) @ (1*dim_out)
        Z = input@self.W + np.ones((self.batch,1)) @ self.b 
        return Z
    
    def backward(self, dLdZ):
    # input loss wrt to output
    """
        - dLdI = dLdZ dZdI = dLdZ W.T
        - dLdW = dLdZ dZdW = dLdZ I.T
        - dLdb = dLdZ dZdb = dLdZ 1
    """
    dLdI = dLdZ @ self.W.T
    self.dLdW = dLdZ @self.input.T
    self.dLdb = dLdZ @ np.ones((self.batch,1))
    # self.dLdb = np.sum(dLdZ, axis=0, keepdims=True)
    
    return dLdI

    