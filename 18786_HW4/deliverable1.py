# %%
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
np.random.seed(0)
torch.manual_seed(0)

# %%
# Activations
# - Relu
# - Tanh
# - Identity

def activation(x, function):
    x = torch.tensor(x, dtype=torch.float32)
    
    if function == "relu":
        output = F.relu(x)
    
    elif function == "tanh":
        output = F.tanh(x) 
    
    elif function == "linear": 
        output = x
    else:
        raise ValueError("Unsupported activation function")
    return output.numpy()

# %%
A = np.array([[1.25, .75], [.75, 1.25]])
B = np.array([[1.25, .75], [.75, 1.25]])

# %%
def f_(x, A=A):
    return A@x+ 0
def g_(x, B=B):
    return B@x+ 0

def rnn(x_t, function_name, A=A, B=B):
    f_x = f_(x_t, A)
    x_t_1 = activation(f_x, function_name) 
    y_t = g_(x_t_1, B) 
    return x_t_1, y_t

# %%
# sample 10 points from a 2D standard normal distribution
x = np.random.randn(10,2)
x

# %%

def get_norm_y(x, function_name, n=15):
    norm_y_list = []
    for i in range(n):
        x, y = rnn(x, function_name, A, B)
        y_norm = np.linalg.norm(y)
        norm_y_list.append(y_norm)
    return norm_y_list


def plot_norms(x, function, name):
    for i in range(10):
        element = x[i,:]
        norm_y_list = get_norm_y(element, function)
        x_list = [i for i in range(1, len(norm_y_list)+1)]
        plt.plot(x_list, norm_y_list, label=f"{i+1}th sample")
    plt.title(f"Norm of y for {name} Activation")
    plt.xlabel("t")
    plt.ylabel(r"$||y_t||_2$")
    plt.legend(title="Samples")
    plt.savefig(f"1_norm_y_{name}.png")
    plt.show()


def plot_norm_2(x, function, name):
    for i in range(2):
        element = x[:, i]
        norm_y_list = get_norm_y(element, function)
        x_list = [i for i in range(1, len(norm_y_list)+1)]
        plt.plot(x_list, norm_y_list, label = f'x{i}')
        plt.title(f"Norm of y for {name} Activation")
        plt.xlabel("t")
        plt.ylabel(r"$||y_t||_2$")
        plt.legend(title="Samples")
    plt.savefig(f"2_norm_y_{name}.png")
    plt.show()


# %%
plot_norms(x, 'linear', "Linear")

# %%
plot_norms(x,'relu', "ReLU")

# %%
plot_norms(x, 'tanh', "Tanh")

# %%
# plot_norm_2(x = np.array([1,1]), function = 'linear', name = 'Linear')
# plot_norm_2(x = np.array([1,-1]), function = 'linear', name = 'Linear')

plot_norm_2(x = np.array([[1,1], [1,-1]]), function = 'linear', name = 'Linear')

plot_norm_2(x = np.array([[1,1], [1,-1]]), function = 'relu', name = 'ReLU')

plot_norm_2(x = np.array([[1,1], [1,-1]]), function = 'tanh', name = 'TanH')


