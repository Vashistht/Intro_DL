
import numpy as np

class Criterion(): # using this as the base class for the loss functions
        def __init__(self) -> None:
            pass
        
        # A: pred, Y: label
        def forward(self, A, Y):
            raise NotImplementedError
        
        def backward(self):
            raise NotImplementedError

'''
Already sigmoids the given model output, so use LinearActivation in the last layer of the forward pass to used this loss function
'''

class CrossEntropyLoss(Criterion):
    def forward(self, A, Y):
        self.A = A
        self.Y = Y
        epsilon = 1e-7
        
        self.softmax = np.exp(A - np.max(A, axis=1, keepdims=True))
        self.softmax /= np.sum(self.softmax, axis=1, keepdims=True)
        
        crossentropy = -np.sum(Y * np.log(self.softmax + epsilon)) / A.shape[0]
        
        return crossentropy
    
    def backward(self):
        dLdA = (self.softmax - self.Y) / self.Y.shape[0]
        return dLdA



class L2Loss(Criterion):
    def forward(self, A, Y):
        self.A = A
        self.Y = Y
        mse = np.mean((A - Y) ** 2)
        return mse
    
    def backward(self):
        dLdA = 2 * (self.A - self.Y) / self.A.shape[0]
        return dLdA

