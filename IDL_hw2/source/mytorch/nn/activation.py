import numpy as np

class Activation(): # using this as the base class for the activation functions
    
    def __init__(self):
        self.A = None
    
    def forward(self, A):
        raise NotImplementedError
    
    def backward(self, dLdA):
        raise NotImplementedError
'''
Note: Terms used here/ structure of the code is inspired from Intro do DL taught by Professors Bhiksha Raj
- using his slides and handout for reference of code structure
'''

class LinearActivation(Activation):
    """Implementation of the Linear Activation."""
    def forward(self, A):
        self.A = A
        return self.A

    def backward(self, dLdA):
        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ
        return dLdZ

class ReLU(Activation):
    """Implementation of the ReLU Activation (Rectified Linear Unit)."""
    def forward(self, A):
        self.A = np.maximum(0, A)
        return self.A

    def backward(self, dLdA):
        dAdZ = np.where(self.A > 0, 1, 0)
        dLdZ = dLdA * dAdZ
        return dLdZ

class Sigmoid(Activation):
    """Implementation of the Sigmoid Activation."""
    def forward(self, A):
        self.A = 1 / (1 + np.exp(-A))
        return self.A

    def backward(self, dLdA):
        dAdZ = self.A * (1 - self.A)
        dLdZ = dLdA * dAdZ
        return dLdZ

class Tanh(Activation):
    """Implementation of the TanH Activation (Hyperbolic Tangent)."""
    def forward(self, A):
        self.A = np.tanh(A)
        return self.A

    def backward(self, dLdA):
        dAdZ = (1 - self.A**2)
        dLdZ = dLdA * dAdZ
        return dLdZ
