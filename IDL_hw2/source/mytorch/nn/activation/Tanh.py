import numpy as np

'''
Here we implement the activations, their derivatives for forward and backward pass.
- used copilot for commenting
'''

class TanH:
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

