import numpy as np


class BatchNorm1d:

    def __init__(self, num_features, alpha=0.9):

        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        # Running mean and variance, updated during training, used during
        # inference
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the
        training phase of the problem or the inference phase.
        So see what values you need to recompute when eval is False.
        """
        self.Z = Z
        self.N = self.Z.shape[0]  # TODO
        self.M = (1/self.N)*np.sum(self.Z, axis=0, keepdims=True)  # TODO
        self.V = (1/self.N)*np.sum((self.Z-self.M)**2, axis=0, keepdims=True)  # TODO
        self.broadband_var = np.ones((self.N,1))

        if eval == False:
            # training mode
            self.NZ = (self.Z - self.broadband_var @ self.M)/np.sqrt(self.broadband_var @ self.V + self.eps)  # TODO
            self.BZ = self.broadband_var @ self.BW * self.NZ + self.broadband_var @ self.Bb  # TODO

            self.running_M = self.alpha * self.running_M + (1-self.alpha) * self.M  # TODO
            self.running_V = self.alpha * self.running_V + (1-self.alpha) * self.V  # TODO
        else:
            # inference mode
            self.NZ = (self.Z - self.broadband_var @ self.running_M)/np.sqrt(self.broadband_var @ self.running_V + self.eps) # TODO
            self.BZ = self.broadband_var @ self.BW * self.NZ + self.broadband_var @ self.Bb  # TODO

        return self.BZ

    def backward(self, dLdBZ):

        self.dLdBW = np.sum(dLdBZ * self.NZ, axis=0, keepdims=True)  # TODO
        self.dLdBb = np.sum(dLdBZ, axis=0, keepdims=True)  # TODO

        dLdNZ = dLdBZ * (self.broadband_var @ self.BW)  # TODO
        dLdV = -0.5 * np.sum(dLdNZ * (self.Z - self.broadband_var @ self.M) * (1/np.sqrt(self.broadband_var @ self.V + self.eps)**3), axis=0, keepdims=True)  # TODO
        dNZdM = -self.broadband_var @ (1/np.sqrt(self.V + self.eps)) - 0.5*(self.Z - self.broadband_var @ self.M) * (self.broadband_var @ (1/np.sqrt(self.V + self.eps)**3)) * ((-2/self.N) * self.broadband_var @ np.sum(self.Z - self.broadband_var @ self.M, axis=0, keepdims=True))
        dLdM = np.sum(dLdNZ * dNZdM, axis=0, keepdims=True)  # TODO

        dLdZ = dLdNZ * (self.broadband_var @ (1/np.sqrt(self.V + self.eps))) + (self.broadband_var @ dLdV) * ((2/self.N) * (self.Z - self.M)) + (1/self.N) * dLdM  # TODO

        return dLdZ
