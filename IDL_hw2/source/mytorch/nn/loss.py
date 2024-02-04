import numpy as np


class CrossEntropyLoss:
    
    def forward(self, pred, labels, epsilon=1e-6):
        """Forward pass for cross entropy loss.

        Parameters
        ----------
        pred : np.array
            Softmax label predictions. Should have shape (batch_size, num_classes).
        labels : np.array
            (batch_size, num_classes).

        Returns
        -------
        float
            Cross entropy loss.
        """
        assert(np.shape(pred) == np.shape(labels))
        crossentropy = np.mean(-np.sum(labels * np.log(pred + epsilon), axis=1))
        
        return crossentropy
    
    def backward(self, pred, labels):
        """Backward pass for cross entropy loss.

        Parameters
        ----------
        pred : np.array
            Softmax label predictions. Should have shape (dim, num_classes).
        labels : np.array
            One-hot true labels. Should have shape (dim, num_classes).

        Returns
        -------
        np.array
            Gradient of the cross entropy loss with respect to the input pred.
        """
        assert(np.shape(pred) == np.shape(labels))
        crossentropy_backward = (pred-labels) / labels.shape[0]
        return crossentropy_backward





class L2Loss:
    
    def forward(self, pred, labels, epsilon=1e-6):
        """Forward pass for L2 loss (mean squared error).

        Parameters
        ----------
        pred : np.array
            Predictions. Should have shape (batch_size, num_classes).
        labels : np.array
            True labels. Should have shape (batch_size, num_classes).

        Returns
        -------
        float
            Mean squared error loss.
        """
        assert(np.shape(pred) == np.shape(labels))
        l2_loss = np.sum((pred - labels)**2, axis=1)
        l2_loss /= ( pred.shape[0] * pred.shape[1])        
        
        return l2_loss
    
    def backward(self, pred, labels):
        """Backward pass for L2 loss.

        Parameters
        ----------
        pred : np.array
            Predictions. Should have shape (batch_size, num_classes).
        labels : np.array
            True labels. Should have shape (batch_size, num_classes).

        Returns
        -------
        np.array
            Gradient of the L2 loss with respect to the input pred.
        """
        assert(np.shape(pred) == np.shape(labels))
        l2_loss_backward = 2* (pred-labels)
        l2_loss_backward /= ( labels.shape[0] * labels.shape[1])
        return l2_loss_backward
    