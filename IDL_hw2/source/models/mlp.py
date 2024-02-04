import numpy as np
import matplotlib.pyplot as plt
from mytorch.nn.activation import ReLU, Softmax, Tanh, LinearActivation
from mytorch.nn.initialization import Xavier, He
from mytorch.nn.loss import CrossEntropyLoss, L2Loss
from mytorch.optim import SGD, Adam
from mytorch.nn import Linear
# imports from given file
import numpyNN


# mlp = initialize_mlp ( num_layers , num_width , opt_act , opt_init )
# train_mlp ( mlp , training_data , num_epoch , opt_loss , opt_optim )
# test_mlp ( mlp , test_data , opt_loss )

# class MLP:
#     def __init__(self, num_layers, node_list, opt_list, opt_init):
        
#         self.num_layers = num_layers
#         self.node_list = node_list
#         self.opt_list = opt_list
#         self.opt_init = opt_init
        
#         self.layers = []
        
#         for i in range(self.num_layers):
#             if i == 0:
#                 self.layers.append(Linear(node_list[i], node_list[i+1], initialization=opt_init))
#             else:
#                 self.layers.append(Linear(node_list[i], node_list[i+1], initialization=opt_init))
#                 self.layers.append(opt_list[i-1])


class MLP:
    def __init__(self, node_list, activation_list, opt_init):
        # assert len(node_list) - 1 == len(activation_list), "node_list must be one more than activation_list"
        
        self.node_list = node_list
        self.activation_list = activation_list
        self.opt_init = opt_init
        
        self.layers = self._build_layers()
    
    def _build_layers(self):
        layers = []
        for i in range(len(self.node_list) - 1):
            # Add Linear layer
            linear_layer = Linear(self.node_list[i], self.node_list[i+1], initialization=self.opt_init)
            layers.append(linear_layer)
            
            # Add activation layer based on activation_list
            activation_fn = self._get_activation_fn(self.activation_list[i])
            layers.append(activation_fn)
            
        return layers


    def _get_activation_fn(self, name):
        if name == 'ReLU':
            return ReLU()
        elif (name == 'Softmax') or (name == 'Sigmoid'):
            return Softmax()
        elif name == 'Tanh':
            return Tanh()
        elif name == 'LinearActivation':
            return LinearActivation()
        else:
            print("Invalid activation function")
            return None
    
    def forward(self, input):
        no_layers = len(self.layers)
        for i in range(no_layers):
            input = self.layers[i].forward(input)
        return input

    def backward(self, dLdA):
        no_layers = len(self.layers)
        for i in reversed(range(no_layers)):
            dLdA = self.layers[i].backward(dLdA)
        return dLdA