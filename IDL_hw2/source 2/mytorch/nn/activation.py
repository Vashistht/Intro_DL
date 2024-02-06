
import numpy as np


class Activation():
    
    def __init__(self):
        self.x = None
    
    def forward(self, x, train=True):
        raise NotImplementedError
    
    def backward(self, x):
        raise NotImplementedError
    




###
class LinearActivation(Activation):
    """Numpy implementation of the Linear Activation.
    """
    def forward(self, x, train=True):
        """Forward propogation thorugh Linear.
        Parameters
        ----------
        x : np.array
            Input for this activation function, x_{k-1}.
        Returns
        -------
        np.array
            Output of this activation function x_k = f_k(., x_{k-1}).
        """
        self.x = x
        return x

    def backward(self, x):
        """Derivative of the sigmoid activation function.
        Parameters
        ----------
        x : np.array
            Input for this activation function, x_{k-1}.
        Returns
        -------
        np.array
            Output of the derivative of the sigmoid activation function.
        """
        return np.ones(x.shape)
    
###

###

class ReLU(Activation):
    """Numpy implementation of the ReLU Activation (Rectified Linear Unit).
    """
    def forward(self, x, train=True):
        """Forward propogation thorugh ReLU.
        Parameters
        ----------
        x : np.array
            Input for this activation function, x_{k-1}.
        Returns
        -------
        np.array
            Output of this activation function x_k = f_k(., x_{k-1}).
        """
        self.x = x
        return np.maximum(0, x)

    def backward(self, x):
        """Derivative of the sigmoid activation function.
        Parameters
        ----------
        x : np.array
            Input for this activation function, x_{k-1}.
        Returns
        -------
        np.array
            Output of the derivative of the sigmoid activation function.
        """
        return np.where(x > 0, 1, 0)


###

'''
Here we implement the activations, their derivatives for forward and backward pass.
- used copilot for commenting
'''

import numpy as np

###
class Softmax(Activation):
    
    def forward(self, x, train=True):
        """Forward propogation through softmax.
        Parameters
        ----------
        x : np.array
            Input for this activation function, x_{k-1}.
        Returns
        -------
        np.array
            Output of this activation function x_k = f_k(., x_{k-1}).
        """
        self.x = x
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def backward(self, x):
        """Derivative of the sigmoid activation function.
        Parameters
        ----------
        x : np.array
            Input for this activation function, x_{k-1}.
        Returns
        -------
        np.array
            Output of the derivative of the sigmoid activation function.
        """
        return x * (1 - x)

import numpy as np

'''
Here we implement the activations, their derivatives for forward and backward pass.
- used copilot for commenting
'''

class Tanh(Activation):
    """Numpy implementation of the TanH Activation (Hyperbolic Tangent).
    """
    def forward(self, x, train=True):
        """Forward propogation thorugh TanH.
        Parameters
        ----------
        x : np.array
            Input for this activation function, x_{k-1}.
        Returns
        -------
        np.array
            Output of this activation function x_k = f_k(., x_{k-1}).
        """
        self.x = x
        return np.tanh(x)

    def backward(self, x):
        """Derivative of the sigmoid activation function.
        Parameters
        ----------
        x : np.array
            Input for this activation function, x_{k-1}.
        Returns
        -------
        np.array
            Output of the derivative of the sigmoid activation function.
        """
        return 1 - np.tanh(x)**2

