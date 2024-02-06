import numpy as np
import scipy

class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self, dLdA):

        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ

        return dLdZ


class Sigmoid:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Sigmoid.
    """
    def forward(self, Z):

        self.A = 1 / (1 + np.exp(-Z)) # TODO

        return self.A

    def backward(self, dLdA):

        dAdZ = self.A - self.A * self.A # TODO
        dLdZ = dLdA * dAdZ # TODO

        return dLdZ


class Tanh:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Tanh.
    """
    def forward(self, Z):

        self.A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z)) # TODO

        return self.A

    def backward(self, dLdA):

        dAdZ = 1 - self.A**2 # TODO
        dLdZ = dLdA * dAdZ # TODO

        return dLdZ


class ReLU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on ReLU.
    """
    def forward(self, Z):

        self.A = np.where(Z <= 0, 0, Z) # TODO

        return self.A

    def backward(self, dLdA):

        dAdZ = np.where(self.A <= 0, 0, 1) # TODO
        dLdZ = dLdA * dAdZ # TODO

        return dLdZ

class GELU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on GELU.
    """
    def forward(self, Z):

        self.A = 0.5*Z * (1+scipy.special.erf(Z/np.sqrt(2))) # TODO
        self.Z = Z

        return self.A

    def backward(self, dLdA):

        dAdZ = 0.5*(1+scipy.special.erf(self.Z/np.sqrt(2))) + (self.Z/np.sqrt(2*np.pi)) * np.exp(-self.Z**2/2) # TODO
        dLdZ = dLdA * dAdZ # TODO

        return dLdZ

class Softmax:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Softmax.
    """

    def forward(self, Z):
        """
        Remember that Softmax does not act element-wise.
        It will use an entire row of Z to compute an output element.
        """

        self.A = np.exp(Z) / (np.sum(np.exp(Z), axis=1, keepdims=True) @ np.ones((1,Z.shape[1]))) # TODO

        return self.A
    
    def backward(self, dLdA):

        # Calculate the batch size and number of features
        N = self.A.shape[0] # TODO
        C = self.A.shape[1] # TODO

        # Initialize the final output dLdZ with all zeros. Refer to the writeup and think about the shape.
        dLdZ = np.zeros((N,C)) # TODO

        # Fill dLdZ one data point (row) at a time
        for i in range(N):

            # Initialize the Jacobian with all zeros.
            J = np.zeros((C,C)) # TODO

            # Fill the Jacobian matrix according to the conditions described in the writeup
            for m in range(C):
                for n in range(C):
                    if m == n:
                        J[m,n] = self.A[i,m] * (1-self.A[i,m]) # TODO
                    else:
                        J[m,n] = -self.A[i,m]*self.A[i,n]

            # Calculate the derivative of the loss with respect to the i-th input
            dLdZ[i,:] = dLdA[i,:] @ J # TODO

        return dLdZ