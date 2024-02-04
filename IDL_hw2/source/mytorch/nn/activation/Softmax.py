'''
Here we implement the activations, their derivatives for forward and backward pass.
- used copilot for commenting
'''

import numpy as np

###
class Softmax:
    
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

