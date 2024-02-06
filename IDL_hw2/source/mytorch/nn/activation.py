
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

    def backward(self,dA):
        """Derivative of the linear activation function.
        Parameters
        ----------
        x : np.array
            Input for this activation function, x_{k-1}.
        Returns
        -------
        np.array
            Output of the derivative of the sigmoid activation function.
        """
        return dA* np.ones(self.x.shape)
    
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

    def backward(self, dA):
        """Derivative of the ReLU activation function.
        Parameters
        ----------
        x : np.array
            Input for this activation function, x_{k-1}.
        Returns
        -------
        np.array
            Output of the derivative of the sigmoid activation function.
        """
        return np.where(self.x > 0, 1, 0)


###

'''
Here we implement the activations, their derivatives for forward and backward pass.
- used copilot for commenting
'''

import numpy as np

###
class Sigmoid(Activation):
    
    def forward(self, x, train=True):
        """Forward propagation through the sigmoid activation function.
        Parameters
        ----------
        x : np.array
            Input for this activation function.
        Returns
        -------
        np.array
            Output of the sigmoid activation function.
        """
        self.x = x
        return 1 / (1 + np.exp(-x))

    def backward(self, dA):
        """Derivative of the sigmoid activation function with respect to the input x.
        Parameters
        ----------
        dA : np.array
            The gradient of the loss with respect to the output of the sigmoid function.
        Returns
        -------
        np.array
            Gradient of the loss with respect to the input of the sigmoid function.
        """
        sigmoid = 1 / (1 + np.exp(-self.x))
        return dA * sigmoid * (1 - sigmoid)


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

    def backward(self, dA):
        """Derivative of the Tanh activation function.
        Parameters
        ----------
        x : np.array
            Input for this activation function, x_{k-1}.
        Returns
        -------
        np.array
            Output of the derivative of the sigmoid activation function.
        """
        return dA* (1 - np.tanh(self.x)**2)

