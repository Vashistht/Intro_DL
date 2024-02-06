import numpy as np
from mytorch.nn.activation import ReLU, Sigmoid, Tanh, LinearActivation
from mytorch.nn.initialization import Xavier, He
from mytorch.nn.linear import Linear
from mytorch.optim.optimizer import SGD, Adam
# imports from given file
import numpyNN



class MLP():
    def __init__(self, input_dim, output_dim, hidden_neuron_list, activation_list, opt_init):

        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.hidden_neuron_list = [self.input_dim] + hidden_neuron_list + [self.output_dim] 
        self.activation_list = activation_list
        self.opt_init = opt_init
        
        self.layers = self._build_layers()
        
        # assert len(self.layers) - 1 == len(activation_list), "layer list must be one more than activation_list"
        
    def _build_layers(self):
        layers = []
        for i in range(len(self.hidden_neuron_list) - 1):
            # Add Linear layer
            linear_layer = Linear(self.hidden_neuron_list[i], self.hidden_neuron_list[i+1], initialization=self.opt_init)
            layers.append(linear_layer)
            
            # Add activation layer based on activation_list
            activation_fn = self._get_activation_fn(self.activation_list[i])
            layers.append(activation_fn)
        # print(layers)
        return layers

    

    def _get_activation_fn(self, name):
        if name == 'ReLU':
            return ReLU()
        elif (name == 'Sigmoid'):
            return Sigmoid()
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

    def get_parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters) # append gave error trying this
        return params

    def summary(self):
        print("Model Summary")
        print("-------------")
        total_params = 0
        for i, layer in enumerate(self.layers):
            layer_type = "Linear" if isinstance(layer, Linear) else type(layer).__name__
            if isinstance(layer, Linear):
                params = layer.dim_in * layer.dim_out + layer.dim_out  # weights + biases
                print(f"Layer {i+1}: {layer_type} - Input Dim: {layer.dim_in}, Output Dim: {layer.dim_out}, Parameters: {params}")
            else:
                print(f"Layer {i+1}: {layer_type}")
                params = 0
            total_params += params
        print(f"Total Parameters: {total_params}")

    def set_parameters(self, best_params):
        param_index = 0  # Index for tracking position in best_params list
        for layer in self.layers:
            if isinstance(layer, Linear):  # Only linear layers have parameters
                layer.W = best_params[param_index]['params']
                layer.b = best_params[param_index + 1]['params']
                param_index += 2  # Increment by 2 to move to the next set of weights and biases

