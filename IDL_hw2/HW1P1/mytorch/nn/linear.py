import numpy as np


class Linear:

    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize the weights and biases with zeros
        Checkout np.zeros function.
        Read the writeup to identify the right shapes for all.
        """
        self.W = np.zeros((out_features, in_features))  # TODO
        self.b = np.zeros((out_features, 1))  # TODO

        self.debug = debug

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output Z of linear layer with shape (N, C1)
        Read the writeup for implementation details
        """
        self.A = A  # TODO
        self.N = A.shape[0]  # TODO store the batch size of input
        # Think how will self.Ones helps in the calculations and uncomment below
        # self.Ones = np.ones((self.N,1))
        Z = A @ self.W.T + np.ones((self.N,1)) @ self.b.T  # TODO

        return Z

    def backward(self, dLdZ):

        dLdA = dLdZ @ self.W  # TODO
        self.dLdW = dLdZ.T @ self.A  # TODO
        self.dLdb = dLdZ.T @ np.ones((self.N,1))  # TODO

        if self.debug:
            
            self.dLdA = dLdA

        return dLdA
