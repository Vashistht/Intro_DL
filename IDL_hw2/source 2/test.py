import numpy as np
import matplotlib.pyplot as plt
from mytorch.nn.activation import ReLU, Softmax, Tanh, LinearActivation
from mytorch.nn.initialization import Xavier, He
from mytorch.nn.linear import Linear
from mytorch.nn.loss import CrossEntropyLoss, L2Loss
from mytorch.optim.optimizer import SGD, Adam
from models.mlp import MLP
###



def train_mlp(mlp, x_train, y_train, opt_loss, opt_optim, num_epoch = 20):
    
    assert(x_train.shape[0]== y_train.shape[1]) # "x_train and y_train must have same length"
    index = np.arange(len(x_train))
    train_loss = []
    train_accuracy = [ ]
    
    for epoch in range(num_epoch):
        np.random.shuffle(index)
        train_x = x_train[index]
        train_y = y_train[index]
        
        y_pred = mlp.forward(train_x)
        y_label = np.argmax(train_y, axis=1)

        loss = opt_loss.forward(y_pred, train_y)
        train_loss.append(loss)

        y_pred = np.argmax(y_pred, axis=1) 
        accuracy = np.sum(y_pred == y_label)/len(train_x)
        train_accuracy.append(accuracy )
        
        dLdZ = opt_loss.backward(y_pred, y_label)
        mlp.backward(dLdZ)
        parameters = mlp.get_parameters()
        opt_optim.step(parameters)
        opt_optim.zero_grad()
        
        print(f"Epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}")
    
    train_logs = {"train_loss": train_loss, "train_accuracy": train_accuracy}
    return train_logs



def test_mlp(mlp, x_test, y_test, opt_loss, num_epoch = 20):
    """
    Parameters
    ----------
    Returns
    -------
        [0] Mean test loss.
        [1] Test accuracy.
    """
    assert(x_test.shape[0] == y_test.shape[0]) # "x_test and y_test must have same length"
    
    test_loss = []
    test_accuracy = [ ]
    
    for epoch in range(num_epoch):
        y_pred = mlp.forward(x_test)
        y_label = np.argmax(y_test, axis=1)
        loss = opt_loss.forward(y_pred, y_test)
        y_pred = np.argmax(y_pred, axis=1)
        accuracy = np.sum(y_pred == y_label)/len(x_test)
        test_loss.append(loss)
        test_accuracy.append(accuracy)
        print(f"Epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}")
    
    test_logs = {"test_loss": loss, "test_accuracy": accuracy}
    return test_logs