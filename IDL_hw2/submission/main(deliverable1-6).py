# %%
import numpy as np
import matplotlib.pyplot as plt

from mytorch.nn.activation import ReLU, Sigmoid, Tanh, LinearActivation
from mytorch.nn.initialization import Xavier, He
from mytorch.nn.linear import Linear
from mytorch.nn.loss import CrossEntropyLoss, L2Loss
from mytorch.optim.optimizer import SGD, Adam
from models.mlp import MLP
import numpyNN

# %%
# based on dataset.py from IML HW 6 
def one_hot_encoding(y, num_classes=2):
    one_hot = np.eye(num_classes)[y.astype(int).flatten()]
    return one_hot

# %%
def train_and_test_mlp(mlp, x_train, y_train, x_test, y_test, opt_loss, opt_optim, num_epoch=20, batch_size=32):
    assert x_train.shape[0] == y_train.shape[0], "x_train and y_train must have the same length"
    assert x_test.shape[0] == y_test.shape[0], "x_test and y_test must have the same length"

    train_loss, train_accuracy = [], []
    test_loss, test_accuracy = [], []

    y_train_encoded = one_hot_encoding(y_train)  
    y_test_encoded = one_hot_encoding(y_test)

    for epoch in range(num_epoch):
        # Shuffle training data and labels
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train_encoded[indices]

        batch_losses = []
        batch_accuracies = []

        # mini-batches
        for start_idx in range(0, x_train.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, x_train.shape[0])
            batch_x = x_train_shuffled[start_idx:end_idx]
            batch_y = y_train_shuffled[start_idx:end_idx]

            y_pred_train = mlp.forward(batch_x)
            loss_train = opt_loss.forward(y_pred_train, batch_y)
            batch_losses.append(np.mean(loss_train))

            dLdZ = opt_loss.backward()  # Use correct call for backward computation
            mlp.backward(dLdZ)
            opt_optim.step()
            opt_optim.zero_grad()

            predicted_labels_train = np.argmax(y_pred_train, axis=1)
            true_labels_train = np.argmax(batch_y, axis=1)
            accuracy_train = np.sum(predicted_labels_train == true_labels_train) / len(batch_x)
            batch_accuracies.append(accuracy_train)

        # Compute mean loss and accuracy for the epoch
        epoch_loss = np.mean(batch_losses)
        epoch_accuracy = np.mean(batch_accuracies)
        train_loss.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)

        # Testing (evaluate the model with the current state on the test set)
        mlp_eval = mlp.copy()  # Ensure your MLP class has a proper copy method
        y_pred_test = mlp_eval.forward(x_test)
        loss_test = opt_loss.forward(y_pred_test, y_test_encoded)
        test_loss.append(np.mean(loss_test))

        predicted_labels_test = np.argmax(y_pred_test, axis=1)
        true_labels_test = np.argmax(y_test_encoded, axis=1)
        accuracy_test = np.sum(predicted_labels_test == true_labels_test) / len(x_test)
        test_accuracy.append(accuracy_test)

        print(f"Epoch: {epoch}, Train Loss: {epoch_loss}, Train Accuracy: {epoch_accuracy}, Test Loss: {np.mean(loss_test)}, Test Accuracy: {accuracy_test}")

    logs = {
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    }
    return logs

# %% [markdown]
# # 2: Linearly Separable Dataset

# %%
x_train, y_train, x_test, y_test = numpyNN.sample_data(data_name = 'linear-separable',nTrain=200, nTest=200)

# %%
numpyNN.plot_train_test_data(x_train, y_train, x_test, y_test, 'Linear')

# %%
dim_in, dim_out = x_train.shape[1], 2
hidden_neuron_list = [1]
activation_list = ['ReLU', 'Sigmoid']
opt_init = None
opt_loss = L2Loss()
mlp = MLP(dim_in, dim_out, hidden_neuron_list, activation_list, opt_init)
opt_optim = SGD(mlp)

# %%
# dim_in, dim_out = x_train.shape[1], 2
# hidden_neuron_list = [4,4]
# activation_list = ['ReLU', 'ReLU', 'LinearActivation']
# opt_init = 'xavier'
# opt_loss = CrossEntropyLoss()
# mlp = MLP(dim_in, dim_out, hidden_neuron_list, activation_list, opt_init)
# opt_optim = SGD(mlp, lr_decay=1, decay_iter=30, momentum=0.9)

# %%
mlp.summary()

# %%
logs = train_and_test_mlp(mlp, x_train, y_train, x_test, y_test, opt_loss, opt_optim, num_epoch=100)

# %%
numpyNN.plot_stats(logs, mlp, x_train, y_train, x_test, y_test, '2_linear', '2_linear')

# %% [markdown]
# ---

# %% [markdown]
# # XOR problem

# %%
x_train, y_train, x_test, y_test = numpyNN.sample_data(data_name = 'XOR',nTrain=200, nTest=200)

# %%
numpyNN.plot_train_test_data(x_train, y_train, x_test, y_test, 'XOR')

# %%
dim_in, dim_out = x_train.shape[1], 2
hidden_neuron_list = [2,2]
activation_list = ['ReLU', 'ReLU', 'Sigmoid']
opt_init = 'he'
opt_loss = L2Loss()
mlp = MLP(dim_in, dim_out, hidden_neuron_list, activation_list, opt_init)
opt_optim = SGD(mlp)

# %%
mlp.summary()

# %%
xor_logs = train_and_test_mlp(mlp, x_train, y_train, x_test, y_test, opt_loss, opt_optim, num_epoch=500)

# %%
numpyNN.plot_stats(xor_logs, mlp, x_train, y_train, x_test, y_test, '2_xor_sgd_', '2_xor_sgd_')

# %% [markdown]
# ### Adam, More params

# %%
dim_in, dim_out = x_train.shape[1], 2
hidden_neuron_list = [4,16]
activation_list = ['ReLU', 'ReLU','Sigmoid']
opt_init = 'xavier'
opt_loss = L2Loss()
mlp = MLP(dim_in, dim_out, hidden_neuron_list, activation_list, opt_init)
opt_optim = Adam(mlp)
print(mlp.summary())

# %%
xor_logs = train_and_test_mlp(mlp, x_train, y_train, x_test, y_test, opt_loss, opt_optim, num_epoch=500)

# %%
numpyNN.plot_stats(xor_logs, mlp, x_train, y_train, x_test, y_test, '3_xor_adam_', '3_xor_adam_')

# %% [markdown]
# # 4: Differences in cost function
# ### 1- L2 Loss
# ### 2- CrossEntropyLoss

# %%
x_train, y_train, x_test, y_test = numpyNN.sample_data(data_name = 'circle',nTrain=200, nTest=200)

# %%
numpyNN.plot_train_test_data(x_train, y_train, x_test, y_test, 'circle')

# %%
dim_in, dim_out = x_train.shape[1], 2
hidden_neuron_list = [4,16]
activation_list = ['Tanh', 'Tanh','Sigmoid']
opt_init = 'xavier'
opt_loss = L2Loss()
mlp = MLP(dim_in, dim_out, hidden_neuron_list, activation_list, opt_init)
opt_optim = Adam(mlp)
print(mlp.summary())

# %%
circle_reg_logs = train_and_test_mlp(mlp, x_train, y_train, x_test, y_test, opt_loss, opt_optim, num_epoch=1000)

# %%
numpyNN.plot_stats(circle_reg_logs, mlp, x_train, y_train, x_test, y_test, 'circle_adam_regression', 'circle_adam_regression')

# %%
dim_in, dim_out = x_train.shape[1], 2
hidden_neuron_list = [4,16]
activation_list = ['Tanh', 'Tanh','LinearActivation']
opt_init = 'xavier'
opt_loss = CrossEntropyLoss()
mlp = MLP(dim_in, dim_out, hidden_neuron_list, activation_list, opt_init)
opt_optim = Adam(mlp)
print(mlp.summary())

# %%
circle_class_logs = train_and_test_mlp(mlp, x_train, y_train, x_test, y_test, opt_loss, opt_optim, num_epoch=1000)

# %%
numpyNN.plot_stats(circle_class_logs, mlp, x_train, y_train, x_test, y_test, 'circle_adam_classification', 'circle_adam_classification')

# %% [markdown]
# # 5. Differences in optimizers
# 
# ### 1. Vanilla Gradient descent

# %%
x_train, y_train, x_test, y_test = numpyNN.sample_data(data_name = 'sinusoid',nTrain=200, nTest=200)

# %%
numpyNN.plot_train_test_data(x_train, y_train, x_test, y_test, 'sinusoid')

# %%
dim_in, dim_out = x_train.shape[1], 2
hidden_neuron_list = [4,8, 16, 64]
activation_list = ['ReLU','ReLU', 'ReLU','ReLU','LinearActivation']
opt_init = 'xavier'
opt_loss = CrossEntropyLoss()
mlp = MLP(dim_in, dim_out, hidden_neuron_list, activation_list, opt_init)
opt_optim = SGD(mlp, lr_decay=1, decay_iter=1000, momentum=0, lr=0.01)
print(mlp.summary())

# %%
sinusoid_vanilla_sgd = train_and_test_mlp(mlp, x_train, y_train, x_test, y_test, opt_loss, opt_optim, batch_size=16, num_epoch=500)

# %%
numpyNN.plot_stats(sinusoid_vanilla_sgd, mlp, x_train, y_train, x_test, y_test, 'sinusoid_vanilla_sgd', 'sinusoid_vanilla_sgd')

# %% [markdown]
# ### 2. gradient descent with momentum

# %%
dim_in, dim_out = x_train.shape[1], 2
hidden_neuron_list = [4,8, 16, 64]
activation_list = ['ReLU','ReLU', 'ReLU','ReLU','LinearActivation']
opt_init = 'xavier'
opt_loss = CrossEntropyLoss()
mlp = MLP(dim_in, dim_out, hidden_neuron_list, activation_list, opt_init)
opt_optim = SGD(mlp, lr_decay=1, decay_iter=1000, momentum=0.9, lr=0.01)
print(mlp.summary())

# %% [markdown]
# 

# %%
sinusoid_sgd_mom = train_and_test_mlp(mlp, x_train, y_train, x_test, y_test, opt_loss, opt_optim, num_epoch=1000)

# %%
numpyNN.plot_stats(sinusoid_sgd_mom, mlp, x_train, y_train, x_test, y_test, 'sinusoid_sgd_mom', 'sinusoid_sgd_mom')

# %% [markdown]
# ### 3. ADAM

# %%
dim_in, dim_out = x_train.shape[1], 2
hidden_neuron_list = [4,8, 16, 64]
activation_list = ['ReLU','ReLU', 'ReLU','ReLU','LinearActivation']
opt_init = 'xavier'
opt_loss = CrossEntropyLoss()
mlp = MLP(dim_in, dim_out, hidden_neuron_list, activation_list, opt_init)
opt_optim = Adam(mlp)
print(mlp.summary())

# %%
adam_logs = train_and_test_mlp(mlp, x_train, y_train, x_test, y_test, opt_loss, opt_optim, num_epoch=1000)

# %%
numpyNN.plot_stats(adam_logs, mlp, x_train, y_train, x_test, y_test, 'adam_logs', 'adam_logs')

# %% [markdown]
# ### trying to fit better 

# %%
dim_in, dim_out = x_train.shape[1], 2
hidden_neuron_list = [8, 16, 64, 8]
activation_list = ['ReLU','ReLU', 'ReLU','ReLU','LinearActivation']
opt_init = 'xavier'
opt_loss = CrossEntropyLoss()
mlp = MLP(dim_in, dim_out, hidden_neuron_list, activation_list, opt_init)
opt_optim = SGD(mlp, lr_decay=1, decay_iter=300, momentum=0.9, lr=0.01)
print(mlp.summary())

# %%
sinusoid_sgd_mom_more = train_and_test_mlp(mlp, x_train, y_train, x_test, y_test, opt_loss, opt_optim, num_epoch=1000)

# %%
numpyNN.plot_stats(sinusoid_sgd_mom_more, mlp, x_train, y_train, x_test, y_test, 'sinusoid_sgd_mom_more_paramas_', 'sinusoid_sgd_mom_more_paramas_')

# %%
dim_in, dim_out = x_train.shape[1], 2
hidden_neuron_list = [8, 16, 64, 8]
activation_list = ['ReLU','ReLU', 'ReLU','ReLU','LinearActivation']
opt_init = 'xavier'
opt_loss = CrossEntropyLoss()
mlp = MLP(dim_in, dim_out, hidden_neuron_list, activation_list, opt_init)
opt_optim = Adam(mlp, learning_rate=0.001)
print(mlp.summary())

# %%
adam_more_params_logs = train_and_test_mlp(mlp, x_train, y_train, x_test, y_test, opt_loss, opt_optim, num_epoch=1000)

# %%
numpyNN.plot_stats(adam_more_params_logs, mlp, x_train, y_train, x_test, y_test, 'sinusoid_adam_more_params', 'sinusoid_adam_more_params')

# %%
x_train, y_train, x_test, y_test = numpyNN.sample_data(data_name = 'sinusoid',nTrain=200, nTest=200)

# %%
dim_in, dim_out = x_train.shape[1], 2
hidden_neuron_list = [8, 16, 32, 8]
activation_list = ['ReLU','ReLU', 'ReLU','ReLU','LinearActivation']
opt_init = 'xavier'
opt_loss = CrossEntropyLoss()
mlp = MLP(dim_in, dim_out, hidden_neuron_list, activation_list, opt_init)
opt_optim = Adam(mlp, learning_rate=0.01)
print(mlp.summary())

# %%
adam_more_params_logs = train_and_test_mlp(mlp, x_train, y_train, x_test, y_test, opt_loss, opt_optim, num_epoch=400)

# %%
numpyNN.plot_stats(adam_more_params_logs, mlp, x_train, y_train, x_test, y_test, 'sinusoid_adam-lr1e-2_more_params', 'sinusoid_adam-lr1e-2_more_params')

# %%
dim_in, dim_out = x_train.shape[1], 2
hidden_neuron_list = [8, 16, 32, 8]
activation_list = ['ReLU','ReLU', 'ReLU','ReLU','LinearActivation']
opt_init = 'xavier'
opt_loss = CrossEntropyLoss()
mlp = MLP(dim_in, dim_out, hidden_neuron_list, activation_list, opt_init)
opt_optim = Adam(mlp, lr_decay=1, decay_iter=250, learning_rate=0.001)
print(mlp.summary())
adam_more_params_logs  = train_and_test_mlp(mlp, x_train, y_train, x_test, y_test, opt_loss, opt_optim, num_epoch=400)

# %%
numpyNN.plot_stats(adam_more_params_logs, mlp, x_train, y_train, x_test, y_test, 'sinusoid_adam-lr-1e-3_more_layers', 'sinusoid_adam-lr-1e-3_more_layers')

# %%
dim_in, dim_out = x_train.shape[1], 2
hidden_neuron_list = [8, 16, 32, 8]
activation_list = ['ReLU','ReLU', 'ReLU','ReLU','LinearActivation']
opt_init = 'xavier'
opt_loss = CrossEntropyLoss()
mlp = MLP(dim_in, dim_out, hidden_neuron_list, activation_list, opt_init)
opt_optim = SGD(mlp, lr_decay=1, decay_iter=300, momentum=0.9, lr=0.01)
print(mlp.summary())
adam_more_params_logs  = train_and_test_mlp(mlp, x_train, y_train, x_test, y_test, opt_loss, opt_optim, num_epoch=500)

numpyNN.plot_stats(adam_more_params_logs, mlp, x_train, y_train, x_test, y_test, '5_sinusoid_sgdmom-9_morelayers_', '5_sinusoid_sgdmom-9_morelayers_')

# %%
dim_in, dim_out = x_train.shape[1], 2
hidden_neuron_list = [8, 16, 32, 8]
activation_list = ['ReLU','ReLU', 'ReLU','ReLU','LinearActivation']
opt_init = 'xavier'
opt_loss = CrossEntropyLoss()
mlp = MLP(dim_in, dim_out, hidden_neuron_list, activation_list, opt_init)
opt_optim = SGD(mlp, lr_decay=1, decay_iter=300, momentum=0, lr=0.01)
print(mlp.summary())
adam_more_params_logs  = train_and_test_mlp(mlp, x_train, y_train, x_test, y_test, opt_loss, opt_optim, num_epoch=500)

numpyNN.plot_stats(adam_more_params_logs, mlp, x_train, y_train, x_test, y_test, '5_sinusoid_sgd_morelayers_', '5_sinusoid_sgd_morelayers_')

# %% [markdown]
# # Swiss Roll

# %%
x_train, y_train, x_test, y_test = numpyNN.sample_data(data_name = 'swiss-roll',nTrain=200, nTest=200)

# %%
numpyNN.plot_train_test_data(x_train, y_train, x_test, y_test, 'swiss-roll')

# %%
# dim_in, dim_out = x_train.shape[1], 2
# hidden_neuron_list = [8, 16, 64, 16]
# activation_list = ['ReLU','ReLU', 'ReLU','ReLU','LinearActivation']
# opt_init = 'xavier'
# opt_loss = CrossEntropyLoss()
# mlp = MLP(dim_in, dim_out, hidden_neuron_list, activation_list, opt_init)
# opt_optim = Adam(mlp, learning_rate=0.01)
# print(mlp.summary())

# %%
dim_in, dim_out = x_train.shape[1], 2
hidden_neuron_list = [8, 16, 64, 12]
activation_list = ['ReLU','ReLU', 'ReLU','ReLU','LinearActivation']
opt_init = 'xavier'
opt_loss = CrossEntropyLoss()
mlp = MLP(dim_in, dim_out, hidden_neuron_list, activation_list, opt_init)
opt_optim = Adam(mlp, learning_rate=0.01)
print(mlp.summary())

# %%
adam_swiss_logs = train_and_test_mlp(mlp, x_train, y_train, x_test, y_test, opt_loss, opt_optim, num_epoch=500)

# %%
numpyNN.plot_stats(adam_swiss_logs, mlp, x_train, y_train, x_test, y_test, 'spiral_adam_more_layers', 'spiral_adam_more_layers')

# %%
dim_in, dim_out = x_train.shape[1], 2
hidden_neuron_list = [8,4]
activation_list = ['ReLU','ReLU','Tanh']
opt_init = 'xavier'
opt_loss = L2Loss()
mlp = MLP(dim_in, dim_out, hidden_neuron_list, activation_list, opt_init)
opt_optim = Adam(mlp, learning_rate=.01)
print(mlp.summary())

# %%
adam_logs = train_and_test_mlp(mlp, x_train, y_train, x_test, y_test, opt_loss, opt_optim, num_epoch=500)

# %%
numpyNN.plot_stats(adam_swiss_logs, mlp, x_train, y_train, x_test, y_test, 'spiral_noembed-loss', 'spiral_noembed-decisionboudary')

# %% [markdown]
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt

from mytorch.nn.activation import ReLU, Sigmoid, Tanh, LinearActivation
from mytorch.nn.initialization import Xavier, He
from mytorch.nn.linear import Linear
from mytorch.nn.loss import CrossEntropyLoss, L2Loss
from mytorch.optim.optimizer import SGD, Adam
from models.mlp import MLP
import numpyNN

# %%
# based on dataset.py from IML HW 6 
def one_hot_encoding(y, num_classes=2):
    one_hot = np.eye(num_classes)[y.astype(int).flatten()]
    return one_hot

# %%
def train_and_test_mlp(mlp, x_train, y_train, x_test, y_test, opt_loss, opt_optim, num_epoch=20, batch_size=32):
    assert x_train.shape[0] == y_train.shape[0], "x_train and y_train must have the same length"
    assert x_test.shape[0] == y_test.shape[0], "x_test and y_test must have the same length"

    train_loss, train_accuracy = [], []
    test_loss, test_accuracy = [], []

    y_train_encoded = one_hot_encoding(y_train)  
    y_test_encoded = one_hot_encoding(y_test)

    for epoch in range(num_epoch):
        # Shuffle training data and labels
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train_encoded[indices]

        batch_losses = []
        batch_accuracies = []

        # mini-batches
        for start_idx in range(0, x_train.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, x_train.shape[0])
            batch_x = x_train_shuffled[start_idx:end_idx]
            batch_y = y_train_shuffled[start_idx:end_idx]

            y_pred_train = mlp.forward(batch_x)
            loss_train = opt_loss.forward(y_pred_train, batch_y)
            batch_losses.append(np.mean(loss_train))

            dLdZ = opt_loss.backward()  # Use correct call for backward computation
            mlp.backward(dLdZ)
            opt_optim.step()
            opt_optim.zero_grad()

            predicted_labels_train = np.argmax(y_pred_train, axis=1)
            true_labels_train = np.argmax(batch_y, axis=1)
            accuracy_train = np.sum(predicted_labels_train == true_labels_train) / len(batch_x)
            batch_accuracies.append(accuracy_train)

        # Compute mean loss and accuracy for the epoch
        epoch_loss = np.mean(batch_losses)
        epoch_accuracy = np.mean(batch_accuracies)
        train_loss.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)

        # Testing (evaluate the model with the current state on the test set)
        mlp_eval = mlp.copy()  # Ensure your MLP class has a proper copy method
        y_pred_test = mlp_eval.forward(x_test)
        loss_test = opt_loss.forward(y_pred_test, y_test_encoded)
        test_loss.append(np.mean(loss_test))

        predicted_labels_test = np.argmax(y_pred_test, axis=1)
        true_labels_test = np.argmax(y_test_encoded, axis=1)
        accuracy_test = np.sum(predicted_labels_test == true_labels_test) / len(x_test)
        test_accuracy.append(accuracy_test)

        print(f"Epoch: {epoch}, Train Loss: {epoch_loss}, Train Accuracy: {epoch_accuracy}, Test Loss: {np.mean(loss_test)}, Test Accuracy: {accuracy_test}")

    logs = {
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    }
    return logs

# %%
def plot_3d (x_train_augmented, y_train, feature, savefig=False, name= None):
    # Splitting x_train_augmented into separate features for easier handling
    feature_1 = x_train_augmented[:, 0]
    feature_2 = x_train_augmented[:, 1]
    feature_3 = x_train_augmented[:, 2]

    # Separating data points based on y_train values
    class_0_indices = (y_train.flatten() == 0)
    class_1_indices = (y_train.flatten() == 1)

    # Plotting Feature 1 vs Feature 3
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1) # (rows, columns, panel number)
    plt.scatter(feature_1[class_0_indices], feature_3[class_0_indices], c='red', label='Class 0')
    plt.scatter(feature_1[class_1_indices], feature_3[class_1_indices], c='blue', label='Class 1')
    feature = '$' + feature + '$'
    title  = f'{feature} vs $x_i$'
    plt.title(rf'{title}')
    plt.xlabel(r'$x_i$')
    plt.ylabel(rf'{feature}')
    plt.legend()

    # Plotting Feature 2 vs Feature 3
    plt.subplot(1, 2, 2)
    plt.scatter(feature_2[class_0_indices], feature_3[class_0_indices], c='red', label='Class 0')
    plt.scatter(feature_2[class_1_indices], feature_3[class_1_indices], c='blue', label='Class 1')
    title  = f'{feature} vs $y_i$'
    plt.title(rf'{title}')    
    plt.xlabel(r'$y_i$')
    plt.ylabel(r'$x_{i}^2 + y_i^2$')
    plt.legend()

    plt.tight_layout()
    if savefig:
        name = name + '-embedding.png'
        plt.savefig(name)
        
    plt.show()

# %%
def make_nonlinearity(f1, f2, non_linear):
    f1, f2 = np.array(f1), np.array(f2)
    # Apply the non-linear function to f1 and f2
    non_linearity = non_linear(f1, f2)
    return non_linearity

# %%
def predict(X, model): # added this to work with my mlp
        y_pred = model.forward(X)
        return np.argmax(y_pred, axis=1)

# %%
def decision_boundary_plot(X, y, non_linearities, model,i=2,levels=1, savefig=False, name=None):
    feature_1 = X[:, 0]
    feature_2 = X[:, 1]
    feature_i = X[:, i]
    # Create a mesh grid of points
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, num=50),
                        np.linspace(y_min, y_max, num=50))
    
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    X_grid_augmented = grid_points
    
    for non_linear in non_linearities:
        non_linearity = make_nonlinearity(xx.ravel(), yy.ravel(), non_linear)
        X_grid_augmented = np.c_[X_grid_augmented, non_linearity]
    
    Z_pred_grid = predict(X_grid_augmented, model=model)    
    Z_pred_reshaped = Z_pred_grid.reshape(xx.shape)
    
    
    fig = plt.figure(figsize=(20, 10))
    ax2 = fig.add_subplot(122, projection='3d')
    #boundary_level = 0  # This is the decision threshold
    ax2.contourf(xx, yy, Z_pred_reshaped, levels = levels, colors=['purple', 'yellow'], alpha=0.4, offset=.0)
    
    # Scatter plot of actual data points colored by their true labels
    scatter = ax2.scatter(feature_1, feature_2, feature_i, c=y, cmap='viridis', edgecolors='k', s=50, alpha=0.6, marker='o')

    ax2.set_title('Predicted Decision Boundary', fontsize=18)
    ax2.set_xlabel('Feature 1', fontsize=15)
    ax2.set_ylabel('Feature 2', fontsize=15)
    ax2.set_zlabel('Feature 3', fontsize=15)
    # ax2.view_init(elev=30, azim=120, vertical_axis='y')
    if savefig:
        name = name + '-decision_boundary.png'
        plt.savefig(name)
    plt.show()

# %% [markdown]
# # 7: Non-linear embeddings: Circle

# %%
x_train, y_train, x_test, y_test = numpyNN.sample_data(data_name = 'circle',nTrain=200, nTest=200)

# %%
numpyNN.plot_train_test_data(x_train, y_train, x_test, y_test, 'circle-no_embedding')

# %%
radial_distance_train = make_nonlinearity(x_train[:,0], x_train[:,1], lambda x, y: (x**2 + y**2))
radial_distance_test = make_nonlinearity(x_test[:,0], x_test[:,1], lambda x, y: (x**2 + y**2))

# %% [markdown]
# GD: x (200000000000,3), y(200000000000,1)
# model.forward(x)
# loss..
# model.backward(dLdZ)
# 
# Minibatch: 200000000000 into sets of 32
# loop over i x[16i:16(i+1)]
# 
# SGD: random point out of 20000000, (gradient_point1)

# %%
x_train_augmented = np.c_[x_train, radial_distance_train]
x_test_augmented = np.c_[x_test, radial_distance_test]

# %%
# plt.scatter(x_train_augmented[:, 0], x_train_augmented[:, 2], c=y_train.flatten())
plot_3d(x_train_augmented, y_train, 'x_{i}^2 + y_i^2')
plot_3d(x_test_augmented,y_test,  'x_{i}^2 + y_i^2')

# %%
feature_1 = x_train_augmented[:, 0]
feature_2 = x_train_augmented[:, 1]
feature_3 = x_train_augmented[:, 2]

fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot using feature_1, feature_2, feature_3 and coloring by y_train
scatter = ax.scatter(feature_1, feature_3, feature_2, c=y_train, cmap='coolwarm', edgecolors='k', s=30, alpha=0.9, marker='o')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.set_title('3D Scatter Plot of Augmented Features')
ax.view_init(elev=25, azim=10, vertical_axis='z')

# Adding a color bar to interpret the colors
fig.colorbar(scatter, shrink=0.5, aspect=5, label='Class Label')

# %%
dim_in, dim_out = 3, 2
hidden_neuron_list = [1]
activation_list = ['ReLU', 'Sigmoid']
opt_init = 'xavier'
opt_loss = L2Loss()
mlp = MLP(dim_in, dim_out, hidden_neuron_list, activation_list, opt_init)
opt_optim = Adam(mlp)
print(mlp.summary())

# %%
circle_embedded_logs = train_and_test_mlp(mlp, x_train_augmented, y_train, x_test_augmented, y_test, opt_loss, opt_optim, num_epoch=750)

# %%
epochs = len(circle_embedded_logs['train_loss'])
plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
numpyNN.plot_loss(circle_embedded_logs)
plt.subplot(1, 2, 2)
numpyNN.plot_accuracy(circle_embedded_logs)
plt.suptitle(f'Loss-Accuracy Plots (for {epochs} epochs)', fontsize=20)
plt.tight_layout()

# %%
non_linearities =[(lambda x, y: (x**2 + y**2))]
decision_boundary_plot(x_test_augmented, y_test, non_linearities, model=mlp,i=2, savefig=True,levels=10, name='cicular-embedding-db')

# %% [markdown]
# # XOR

# %%
x_train, y_train, x_test, y_test = numpyNN.sample_data(data_name = 'XOR',nTrain=200, nTest=200)

# %%
numpyNN.plot_train_test_data(x_train, y_train, x_test, y_test, 'XOR')

# %%
# Now, separate x_train data points
x_train_0 = x_train[y_train.flatten() == 0]
x_train_1 = x_train[y_train.flatten() == 1]

plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.scatter(x_train_0[:, 0], x_train_0[:, 1], c='red', label='0')
plt.scatter(x_train_1[:, 0], x_train_1[:, 1], c='blue', label=' 1')
plt.legend()
# vertical line at 0
plt.axvline(x=0, color='k', linestyle='--')
# horizontal line at 0
plt.axhline(y=0, color='k', linestyle='--')

# %%
feature_3 = x_train[:, 0] * x_train[:, 1]
feature_3_0 = feature_3[y_train.flatten() == 0]
feature_3_1 = feature_3[y_train.flatten() == 1]

plt.scatter(np.linspace(-1, 1, num=len(feature_3_0))
, feature_3_0, c='red', label='0')

plt.scatter(np.linspace(-1, 1, num=len(feature_3_1))
,feature_3_1, c='blue', label=' 1')
plt.hlines(y=0, xmin=-1, xmax=1, color='k', linestyle='--')
plt.title('Feature 3 (Train)')
plt.legend()

# %%
radial_distance_train = make_nonlinearity(x_train[:,0], x_train[:,1], lambda x, y: (x*y))
radial_distance_test = make_nonlinearity(x_test[:,0], x_test[:,1], lambda x, y: (x*y))

# %%
x_train_augmented = np.c_[x_train, radial_distance_train]
x_test_augmented = np.c_[x_test, radial_distance_test]

# %%
# plt.scatter(x_train_augmented[:, 0], x_train_augmented[:, 2], c=y_train.flatten())
plot_3d(x_train_augmented, y_train, 'x_{i} * y_i')
plot_3d(x_test_augmented,y_test,  'x_{i} * y_i')

# %%
feature_1 = x_train_augmented[:, 0]
feature_2 = x_train_augmented[:, 1]
feature_3 = x_train_augmented[:, 2]

fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot using feature_1, feature_2, feature_3 and coloring by y_train
scatter = ax.scatter(feature_1, feature_3, feature_2, c=y_train, cmap='coolwarm', edgecolors='k', s=30, alpha=0.9, marker='o')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.set_title('3D Scatter Plot of Augmented Features')
ax.view_init(elev=25, azim=10, vertical_axis='z')

# Adding a color bar to interpret the colors
fig.colorbar(scatter, shrink=0.5, aspect=5, label='Class Label')

# %%
dim_in, dim_out = 3, 2
hidden_neuron_list = [1]
activation_list = ['ReLU', 'Sigmoid']
opt_init = 'xavier'
opt_loss = L2Loss()
mlp = MLP(dim_in, dim_out, hidden_neuron_list, activation_list, opt_init)
opt_optim = Adam(mlp)
print(mlp.summary())

# %%
xor_embedded_logs = train_and_test_mlp(mlp, x_train_augmented, y_train, x_test_augmented, y_test, opt_loss, opt_optim, num_epoch=1000)

# %%
epochs = len(xor_embedded_logs['train_loss'])
plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
numpyNN.plot_loss(xor_embedded_logs)
plt.subplot(1, 2, 2)
numpyNN.plot_accuracy(xor_embedded_logs)
plt.suptitle(f'Loss-Accuracy Plots (for {epochs} epochs)', fontsize=20)
plt.savefig('xor-embedding-train-loss.png')
plt.tight_layout()

# %%
non_linearities =[(lambda x, y: (x*y))]
decision_boundary_plot(x_test_augmented, y_test, non_linearities, model=mlp,i=2, savefig=True,levels=1, name='xor-embedding-db')

# %% [markdown]
# # Swiss Roll

# %%
x_train, y_train, x_test, y_test = numpyNN.sample_data(data_name = 'swiss-roll',nTrain=200, nTest=200)

# %%
numpyNN.plot_train_test_data(x_train, y_train, x_test, y_test, 'XOR')

# %%
xsq_train = make_nonlinearity(x_train[:,0], x_train[:,1], lambda x, y: x**2 )
xsq_test = make_nonlinearity(x_test[:,0], x_test[:,1], lambda x, y: x**2 )
nonlin1 = lambda x, y: x**2

ysq_train = make_nonlinearity(x_train[:,0], x_train[:,1], lambda x, y: y**2 ) # np.sqrt(x**2 + y**2)
ysq_test = make_nonlinearity(x_test[:,0], x_test[:,1], lambda x, y: y**2 )
nonlin2 = lambda x, y: y**2

xy_train = make_nonlinearity(x_train[:,0], x_train[:,1], lambda x, y: x*y )
xy_test = make_nonlinearity(x_test[:,0], x_test[:,1], lambda x, y: x*y )
nonlin3 = lambda x, y: x*y

# theta_train = make_nonlinearity(x_train[:,0], x_train[:,1], lambda x, y: np.arctan(y/x) )
# theta_test = make_nonlinearity(x_test[:,0], x_test[:,1], lambda x, y:  np.arctan(y/x) )
# nonlin4 = lambda x, y:  np.arctan(y/x)

# %%
# plt.scatter(radial_distance_train,theta_train,c=y_train.flatten())

# %%
radial_distance_train = make_nonlinearity(x_train[:,0], x_train[:,1], lambda x, y: (x**2 + y**2))
radial_distance_test = make_nonlinearity(x_test[:,0], x_test[:,1], lambda x, y: (x**2 + y**2))

# %%
x_train_augmented = np.c_[x_train, xsq_train, ysq_train, xy_train] # , theta_train, ]
x_test_augmented = np.c_[x_test, xsq_test, ysq_test, xy_test] # , theta_test, ]

# %%
x_train_augmented.shape, x_test_augmented.shape

# %%
dim_in, dim_out = 5, 2
hidden_neuron_list = [8,4]
activation_list = ['ReLU','ReLU','Tanh']
opt_init = 'xavier'
opt_loss = L2Loss()
mlp = MLP(dim_in, dim_out, hidden_neuron_list, activation_list, opt_init)
opt_optim = Adam(mlp, learning_rate=.01)
print(mlp.summary())

# %%
spiral_embedded_logs = train_and_test_mlp(mlp, x_train_augmented, y_train, x_test_augmented, y_test, opt_loss, opt_optim, num_epoch=500)

# %%
feature_1 = x_train_augmented[:, 0]
feature_2 = x_train_augmented[:, 1]
feature_3 = x_train_augmented[:, 2]

fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot using feature_1, feature_2, feature_3 and coloring by y_train
scatter = ax.scatter(feature_1,feature_2,  feature_3, c=y_train, cmap='coolwarm', edgecolors='k', s=30, alpha=0.9, marker='o')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.set_title('3D Scatter Plot of Augmented Features')
ax.view_init(elev=30, azim=120, vertical_axis='y')

# Adding a color bar to interpret the colors
fig.colorbar(scatter, shrink=0.5, aspect=5, label='Class Label')

# %%
non_linearities =[nonlin1, nonlin2, nonlin3]
decision_boundary_plot(x_test_augmented, y_test, non_linearities, model=mlp,i=3, savefig=True,levels=1, name='spiral-embedding-db')

# %%
numpyNN.plot_stats(adam_swiss_logs, mlp, x_train, y_train, x_test, y_test, 'adam_swiss_logs', 'adam_swiss_logs')

# %%
dim_in, dim_out = x_train.shape[1], 2
hidden_neuron_list = [8, 16, 64, 16]
activation_list = ['ReLU','ReLU', 'ReLU','ReLU','LinearActivation']
opt_init = 'xavier'
opt_loss = CrossEntropyLoss()
mlp = MLP(dim_in, dim_out, hidden_neuron_list, activation_list, opt_init)
opt_optim = Adam(mlp, learning_rate=0.01)
print(mlp.summary())

# %%
adam_swiss_logs = train_and_test_mlp(mlp, x_train, y_train, x_test, y_test, opt_loss, opt_optim, num_epoch=500, batch_size=64)
numpyNN.plot_stats(adam_swiss_logs, mlp, x_train, y_train, x_test, y_test, 'adam_swiss_batch64', 'adam_swiss_batch64')

# %% [markdown]
# # 7: Non-linear embeddings

# %%
x_train, y_train, x_test, y_test = numpyNN.sample_data(data_name = 'circle',nTrain=200, nTest=200)

# %%
numpyNN.plot_train_test_data(x_train, y_train, x_test, y_test, 'circle-no_embedding')

# %%
x_train_nonlinear = x_train**2
x_test_nonlinear = x_test**2
x_train_nonlinear = x_train_nonlinear [:, 0] + x_train_nonlinear [:, 1]
x_test_nonlinear = x_test_nonlinear [:, 0] + x_test_nonlinear [:, 1]

# %%
third_feature_train = np.sum(x_train**2, axis=1)
x_train_augmented = np.concatenate((x_train, third_feature_train[:, np.newaxis]), axis=1)
third_feature_test = np.sum(x_train**2, axis=1)
x_test_augmented = np.concatenate((x_train, third_feature_test[:, np.newaxis]), axis=1)

# %%
import matplotlib.pyplot as plt
import numpy as np

# Splitting x_train_augmented into separate features for easier handling
feature_1 = x_train_augmented[:, 0]
feature_2 = x_train_augmented[:, 1]
feature_3 = x_train_augmented[:, 2]

# Separating data points based on y_train values
class_0_indices = (y_train.flatten() == 0)
class_1_indices = (y_train.flatten() == 1)

# Plotting Feature 1 vs Feature 3
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1) # (rows, columns, panel number)
plt.scatter(feature_1[class_0_indices], feature_3[class_0_indices], c='red', label='Class 0')
plt.scatter(feature_1[class_1_indices], feature_3[class_1_indices], c='blue', label='Class 1')
plt.title(r'$x_i$ vs $x_{i}^2 + y_i^2$')
plt.ylabel(r'$y_i$')
plt.ylabel(r'$x_{i}^2 + y_i^2$')
plt.legend()

# Plotting Feature 2 vs Feature 3
plt.subplot(1, 2, 2)
plt.scatter(feature_2[class_0_indices], feature_3[class_0_indices], c='red', label='Class 0')
plt.scatter(feature_2[class_1_indices], feature_3[class_1_indices], c='blue', label='Class 1')
plt.title(r'$y_i$ vs $x_{i}^2 + y_i^2$')
plt.ylabel(r'$x_i$')
plt.ylabel(r'$x_{i}^2 + y_i^2$')
plt.legend()

plt.tight_layout()
plt.show()


# %%
dim_in, dim_out = 3, 2
hidden_neuron_list = [2,4]
activation_list = ['ReLU', 'ReLU', 'Sigmoid']
opt_init = 'xavier'
opt_loss = CrossEntropyLoss()
mlp = MLP(dim_in, dim_out, hidden_neuron_list, activation_list, opt_init)
opt_optim = Adam(mlp)
print(mlp.summary())

# %%
circle_embedded_logs = train_and_test_mlp(mlp, x_train_augmented, y_train, x_test_augmented, y_test, opt_loss, opt_optim, num_epoch=200)

# %%
epochs = len(circle_embedded_logs['train_loss'])
plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
numpyNN.plot_loss(circle_embedded_logs)
plt.subplot(1, 2, 2)
numpyNN.plot_accuracy(circle_embedded_logs)
plt.suptitle(f'Loss-Accuracy Plots (for {epochs} epochs)', fontsize=20)
plt.tight_layout()

# %%
def predict(X, model): # added this to work with my mlp
        y_pred = model.forward(X)
        return np.argmax(y_pred, axis=1)

# %%
X = x_test_augmented    
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1


xx, yy = np.meshgrid(np.linspace(x_min, x_max, num=50),
                        np.linspace(y_min, y_max, num=50))

Z_feature = np.square(xx.ravel()) + np.square(yy.ravel())

X_augmented = np.c_[xx.ravel(), yy.ravel(), Z_feature]

Z_pred = predict(X_augmented, model=mlp)

# %%
# Now, let's plot using the simulated Z_pred
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(121, projection='3d')

# Since Z_pred represents class predictions, we plot them in 3D space
scatter = ax1.scatter(xx.ravel(), yy.ravel(), Z_feature, c=Z_pred, cmap='coolwarm', depthshade=True, alpha=0.1, label='Predicted Class')

ax1.scatter(feature_1, feature_2 , feature_3, c=y_train, cmap='coolwarm', edgecolors='k', s=30, alpha=0.8, label='Data Points', depthshade=True)

ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.set_zlabel('Predicted Class')
ax1.legend()
fig.colorbar(scatter, shrink=0.5, aspect=5)
plt.title('Simulated 3D Decision Boundary Visualization')


# Now, let's plot using the simulated Z_pred
ax2 = fig.add_subplot(122, projection='3d')

# Since Z_pred represents class predictions, we plot them in 3D space
scatter = ax2.scatter(xx.ravel(), yy.ravel(), Z_pred.reshape(xx.shape), c=Z_pred, cmap='coolwarm', depthshade=True)

ax2.scatter(feature_1, feature_2 , feature_3, c=y_train, cmap='coolwarm', edgecolors='k', s=30, alpha=0.3, label='Data Points',depthshade=True)
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.set_zlabel('Predicted Class')

fig.colorbar(scatter, shrink=0.5, aspect=5)
plt.title('Simulated 3D Decision Boundary Visualization')
plt.show()

# %%


# %%


# %%


# %%


# %%
# def train_and_test_mlp_regression(mlp, x_train, y_train, x_test, y_test, opt_loss, opt_optim, num_epoch=20, batch_size=32):
#     assert x_train.shape[0] == y_train.shape[0], "x_train and y_train must have the same length"
#     assert x_test.shape[0] == y_test.shape[0], "x_test and y_test must have the same length"

#     train_loss = []
#     test_loss = []

  
#     for epoch in range(num_epoch):
#         # Shuffle training data and labels
#         indices = np.arange(x_train.shape[0])
#         np.random.shuffle(indices)
#         x_train_shuffled = x_train[indices]
#         y_train_shuffled = y_train[indices]

#         batch_losses = []

#         # mini-batches
#         for start_idx in range(0, x_train.shape[0], batch_size):
#             end_idx = min(start_idx + batch_size, x_train.shape[0])
#             batch_x = x_train_shuffled[start_idx:end_idx]
#             batch_y = y_train_shuffled[start_idx:end_idx]

#             y_pred_train = mlp.forward(batch_x)
#             loss_train = opt_loss.forward(y_pred_train, batch_y)
#             batch_losses.append(np.mean(loss_train))

#             dLdZ = opt_loss.backward()  # Use correct call for backward computation
#             mlp.backward(dLdZ)
#             opt_optim.step()
#             opt_optim.zero_grad()

#         # Compute mean loss and accuracy for the epoch
#         epoch_loss = np.mean(batch_losses)
#         train_loss.append(epoch_loss)

#         # Testing (evaluate the model with the current state on the test set)
#         mlp_eval = mlp.copy()  # Ensure your MLP class has a proper copy method
#         y_pred_test = mlp_eval.forward(x_test)
#         loss_test = opt_loss.forward(y_pred_test, y_test)
#         test_loss.append(np.mean(loss_test))


#         print(f"Epoch: {epoch}, Train Loss: {epoch_loss}, Test Loss: {np.mean(loss_test)}")

#     logs = {
#         "train_loss": train_loss,
#         "test_loss": test_loss,
#     }
#     return logs

# %%
# def train_and_test_scalar(mlp, x_train, y_train, x_test, y_test, opt_loss, opt_optim, num_epoch=20, batch_size=32):
#     assert x_train.shape[0] == y_train.shape[0], "x_train and y_train must have the same length"
#     assert x_test.shape[0] == y_test.shape[0], "x_test and y_test must have the same length"

#     train_loss, train_accuracy = [], []
#     test_loss, test_accuracy = [], []

#     # Assume y_train and y_test are already appropriate for a binary classification task
#     # i.e., they are not one-hot encoded but are binary labels

#     for epoch in range(num_epoch):
#         # Shuffle training data and labels
#         indices = np.arange(x_train.shape[0])
#         np.random.shuffle(indices)
#         x_train_shuffled = x_train[indices]
#         y_train_shuffled = y_train[indices]

#         batch_losses = []
#         batch_accuracies = []

#         # Mini-batches
#         for start_idx in range(0, x_train.shape[0], batch_size):
#             end_idx = min(start_idx + batch_size, x_train.shape[0])
#             batch_x = x_train_shuffled[start_idx:end_idx]
#             batch_y = y_train_shuffled[start_idx:end_idx]

#             y_pred_train = mlp.forward(batch_x).flatten()  # Flatten the output if it's not already 1D
#             loss_train = opt_loss.forward(y_pred_train, batch_y)
#             batch_losses.append(np.mean(loss_train))

#             dLdZ = opt_loss.backward(y_pred_train, batch_y)  # Pass predictions and true values
#             mlp.backward(dLdZ)
#             opt_optim.step()
#             opt_optim.zero_grad()

#             predictions_train = (y_pred_train > 0.5).astype(int)  # Convert probabilities to binary predictions
#             accuracy_train = np.mean(predictions_train == batch_y)
#             batch_accuracies.append(accuracy_train)

#         # Compute mean loss and accuracy for the epoch
#         epoch_loss = np.mean(batch_losses)
#         epoch_accuracy = np.mean(batch_accuracies)
#         train_loss.append(epoch_loss)
#         train_accuracy.append(epoch_accuracy)

#         # Testing (evaluate the model with the current state on the test set)
#         mlp_eval = mlp.copy()  # Ensure your MLP class has a proper copy method
#         y_pred_test = mlp_eval.forward(x_test).flatten()  # Flatten the output if it's not already 1D
#         loss_test = opt_loss.forward(y_pred_test, y_test)
#         test_loss.append(np.mean(loss_test))

#         predictions_test = (y_pred_test > 0.5).astype(int)  # Convert probabilities to binary predictions
#         accuracy_test = np.mean(predictions_test == y_test)
#         test_accuracy.append(accuracy_test)

#         print(f"Epoch: {epoch}, Train Loss: {epoch_loss}, Train Accuracy: {epoch_accuracy}, Test Loss: {np.mean(loss_test)}, Test Accuracy: {accuracy_test}")

#     logs = {
#         "train_loss": train_loss,
#         "train_accuracy": train_accuracy,
#         "test_loss": test_loss,
#         "test_accuracy": test_accuracy
#     }
#     return logs



