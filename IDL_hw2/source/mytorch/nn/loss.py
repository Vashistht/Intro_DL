# Path: mytorch/nn/initialization.py

import numpy as np

class Criterion():
        def __init__(self) -> None:
            pass
        
        # A: pred, Y: label
        def forward(self, A, Y):
            raise NotImplementedError
        
        def backward(self):
            raise NotImplementedError



class CrossEntropyLoss(Criterion):
    def forward(self, A, Y):
        self.A = A
        self.Y = Y
        epsilon = 1e-7
        
        # Correct calculation of softmax probabilities
        self.softmax = np.exp(A - np.max(A, axis=1, keepdims=True))
        self.softmax /= np.sum(self.softmax, axis=1, keepdims=True)
        
        # Correct calculation of cross entropy loss
        crossentropy = -np.sum(Y * np.log(self.softmax + epsilon)) / A.shape[0]
        
        return crossentropy
    
    def backward(self):
        # Correct gradient calculation for cross entropy loss
        dLdA = (self.softmax - self.Y) / self.Y.shape[0]
        return dLdA




class L2Loss(Criterion):
    def forward(self, A, Y):
        self.A = A
        self.Y = Y
        # Mean squared error calculation
        mse = np.mean((A - Y) ** 2)
        return mse
    
    def backward(self):
        # Gradient of the mean squared error with respect to A
        dLdA = 2 * (self.A - self.Y) / self.A.shape[0]
        return dLdA

