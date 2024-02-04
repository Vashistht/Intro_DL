import numpy as np


###

class ReLU:
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