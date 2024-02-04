
import numpy as np

###
class Linear_Activation:
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
