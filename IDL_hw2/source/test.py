import numpy as np
import matplotlib.pyplot as plt
from mytorch.nn.activation import ReLU, Softmax, Tanh, LinearActivation
from mytorch.nn.initialization import Xavier, He
from mytorch.nn.Linear import Linear
from mytorch.nn.loss import CrossEntropyLoss
from mytorch.optim import SGD, Adam
from models.mlp import MLP


# mlp = MLP([2, 3, 2], ['ReLU', 'Softmax'], 'he')
x = Linear(2, 3, initialization='he')
print(x.W)