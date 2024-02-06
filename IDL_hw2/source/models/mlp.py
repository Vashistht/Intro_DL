import numpy as np
from mytorch.nn.activation import ReLU, Sigmoid, Tanh, LinearActivation
from mytorch.nn.initialization import Xavier, He
from mytorch.nn.linear import Linear
from mytorch.optim.optimizer import SGD, Adam
# imports from given file
import numpyNN



class MLP():
    def __init__(self, input_dim, output_dim, hidden_neuron_list, activation_list, opt_init, debug = False):

        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.hidden_neuron_list = [self.input_dim] + hidden_neuron_list + [self.output_dim] 
        self.activation_list = activation_list
        self.opt_init = opt_init
        
        self.debug = debug
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
    
    # understand debug
    def forward(self, A):
        if self.debug:
            self.A = [A]
        no_layers = len(self.layers)
        for i in range(no_layers):
            A = self.layers[i].forward(A)
            if self.debug:
                self.A.append(A)
        return A

    def backward(self, dLdA):
        if self.debug:
            self.dLdA= [dLdA]
            
        no_layers = len(self.layers)
        for i in reversed(range(no_layers)):
            dLdA = self.layers[i].backward(dLdA)
            if self.debug:
                self.dLdA = [dLdA] + self.dLdA  
        return dLdA


    def summary(self):
        print("Model Summary")
        print("-------------")
        total_params = 0
        for i, layer in enumerate(self.layers):
            layer_type = "Linear" if isinstance(layer, Linear) else type(layer).__name__
            if isinstance(layer, Linear):
                params = layer.dim_in * layer.dim_out + layer.dim_out  # weights + biases
                print(f"Layer {i+1}: {layer_type} - A Dim: {layer.dim_in}, Output Dim: {layer.dim_out}, Parameters: {params}")
            else:
                print(f"Layer {i+1}: {layer_type}")
                params = 0
            total_params += params
        print(f"Total Parameters: {total_params}")

    
    def copy(self):
        # Create a new instance of MLP without initializing the parameters.
        copied_mlp = MLP(self.input_dim, self.output_dim, self.hidden_neuron_list[1:-1], self.activation_list, self.opt_init, self.debug)
        
        # Manually copy over the parameters for each layer.
        for original_layer, copied_layer in zip(self.layers, copied_mlp.layers):
            if isinstance(original_layer, Linear):
                copied_layer.W = original_layer.W.copy()
                copied_layer.b = original_layer.b.copy()
        
        return copied_mlp
    
    # def set_parameters(self, best_params):
    #     param_index = 0  # Index for tracking position in best_params list
    #     for layer in self.layers:
    #         if isinstance(layer, Linear):  # Only linear layers have parameters
    #             layer.W = best_params[param_index]['params']
    #             layer.b = best_params[param_index + 1]['params']
    #             param_index += 2  # Increment by 2 to move to the next set of weights and biases


    # def get_parameters(self):
    #     params = []
    #     for layer in self.layers:
    #         if hasattr(layer, 'parameters'):
    #             params.extend(layer.parameters) # append gave error trying this
    #     return params