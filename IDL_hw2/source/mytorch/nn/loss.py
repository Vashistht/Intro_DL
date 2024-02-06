import numpy as np


class Criterion():
        def __init__(self) -> None:
            pass
        
        # A: pred, Y: label
        def forward(self, A, Y):
            raise NotImplementedError
        
        def backward(self):
            raise NotImplementedError



class CrossEntropyLoss(Criterion):
    
    def forward(self, A, Y):
        """Forward pass for cross entropy loss.

        Parameters
        ----------
        A : np.array
            Softmax label predictions. Should have shape (batch_size, num_classes).
        Y : np.array
            (batch_size, num_classes).

        Returns
        -------
        float
            Cross entropy loss.
        """
        self.A = A
        self.Y = Y
        N = self.A.shape[0] # batch 
        C= self.A.shape[1] # num_classes
    
        assert(np.shape(A) == np.shape(Y))
        epsilon = 1e-7

        self.sigmoid = np.exp(self.A) / np.sum(np.exp(self.A), axis=1, keepdims=True) @np.ones((1,C))
        
        
        crossentropy = -self.Y * np.log(self.softmax+ epsilon) @ np.ones((C,1))
        
        sum_crossentropy = np.ones((1,N)) @ crossentropy     
        loss = sum_crossentropy/N
        return loss[0][0]
        
    def backward(self):
        """Backward pass for cross entropy loss.

        Parameters
        ----------
        A : np.array
            Softmax label predictions. Should have shape (dim, num_classes).
        Y : np.array
            One-hot true Y. Should have shape (dim, num_classes).

        Returns
        -------
        np.array
            Gradient of the cross entropy loss with respect to the input A.
        """        
        dLdA = (self.softmax - self.Y) / self.Y.shape[0]
        return dLdA





class L2Loss(Criterion):
    
    def forward(self, A, Y):
        """Forward pass for L2 loss (mean squared error).

        Parameters
        ----------
        A : np.array
            Predictions. Should have shape (batch_size, num_classes).
        Y : np.array
            True Y. Should have shape (batch_size, num_classes).

        Returns
        -------
        float
            Mean squared error loss.
        """
        assert(np.shape(A) == np.shape(Y))
        self.A = A
        self.Y = Y
        self.N = self.A.shape[0] # batch 
        self.C = self.A.shape[1] # num_classes
    
        assert(np.shape(A) == np.shape(Y))
        epsilon = 1e-7

        squared_error = (A - Y)**2
        mean_squared_error = np.mean(squared_error, axis=1)
        # l2_loss = np.sum((A - Y)**2, axis=1)
        # l2_loss /= ( A.shape[0] )        #* A.shape[1]
        
        return mean_squared_error
    
    def backward(self):
        """Backward pass for L2 loss.

        Parameters
        ----------
        A : np.array
            Predictions. Should have shape (batch_size, num_classes).
        Y : np.array
            True Y. Should have shape (batch_size, num_classes).

        Returns
        -------
        np.array
            Gradient of the L2 loss with respect to the input A.
        """
        # assert(np.shape(A) == np.shape(Y))
        dLdA = 2* (self.A-self.Y)
        dLdA /= (self.N)  #* self.C
        return dLdA
    