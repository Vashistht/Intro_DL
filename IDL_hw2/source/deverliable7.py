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
    assert x_train.shape[0] == y_train.shape[0]
    assert x_test.shape[0] == y_test.shape[0]

    train_loss, train_accuracy = [], []
    test_loss, test_accuracy = [], []

    y_train_encoded = one_hot_encoding(y_train)  
    y_test_encoded = one_hot_encoding(y_test)

    for epoch in range(num_epoch):
        indices = np.arange(x_train.shape[0]) # shuffle
        np.random.shuffle(indices)
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train_encoded[indices]

        batch_losses = []
        batch_accuracies = []

        # batches
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

        # Mean loss and acc
        epoch_loss = np.mean(batch_losses)
        epoch_accuracy = np.mean(batch_accuracies)
        train_loss.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)

        # Testing
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
    feature_1 = x_train_augmented[:, 0]
    feature_2 = x_train_augmented[:, 1]
    feature_3 = x_train_augmented[:, 2]

    class_0_indices = (y_train.flatten() == 0)
    class_1_indices = (y_train.flatten() == 1)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(feature_1[class_0_indices], feature_3[class_0_indices], c='red', label='Class 0')
    plt.scatter(feature_1[class_1_indices], feature_3[class_1_indices], c='blue', label='Class 1')
    feature = '$' + feature + '$'
    title  = f'{feature} vs $x_i$'
    plt.title(rf'{title}')
    plt.xlabel(r'$x_i$')
    plt.ylabel(rf'{feature}')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(feature_2[class_0_indices], feature_3[class_0_indices], c='red', label='Class 0')
    plt.scatter(feature_2[class_1_indices], feature_3[class_1_indices], c='blue', label='Class 1')
    title  = f'{feature} vs $y_i$'
    plt.title(rf'{title}')    
    plt.xlabel(r'$y_i$')
    plt.ylabel(rf'{feature}')
    plt.legend()

    plt.tight_layout()
    if savefig:
        name = name + '-embedding.png'
        plt.savefig(name)
    plt.show()

# %%
def make_nonlinearity(f1, f2, non_linear):
    f1, f2 = np.array(f1), np.array(f2)
    non_linearity = non_linear(f1, f2)
    return non_linearity

# %%
def predict(X, model): # added this to work with my mlp
        y_pred = model.forward(X)
        return np.argmax(y_pred, axis=1)

# %%
# 3d inspired decision boundary plot from numpyNN given decision_boundary function 
def decision_boundary_plot(X, y, non_linearities, model,i=2,levels=10, savefig=False, name=None):
    feature_1 = X[:, 0]
    feature_2 = X[:, 1]
    feature_i = X[:, i]

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
    ax2.contourf(xx, yy, Z_pred_reshaped, levels = levels, colors=['purple', 'yellow'], alpha=0.4, offset=.2)
    
    ax2.scatter(feature_1, feature_2, feature_i, c=y, cmap='viridis', edgecolors='k', s=50, alpha=0.6, marker='o')

    ax2.set_title('Predicted Decision Boundary', fontsize=18)
    ax2.set_xlabel('Feature 1', fontsize=15)
    ax2.set_ylabel('Feature 2', fontsize=15)
    ax2.set_zlabel('Feature 3', fontsize=15)
    # ax2.view_init(elev=30, azim=120, vertical_axis='y')
    if savefig:
        name = name + '-decision_boundary.png'
        plt.savefig(name, dpi=300)
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

# %%
x_train_augmented = np.c_[x_train, radial_distance_train]
x_test_augmented = np.c_[x_test, radial_distance_test]

# %%
# plt.scatter(x_train_augmented[:, 0], x_train_augmented[:, 2], c=y_train.flatten())
plot_3d(x_train_augmented, y_train, 'x_{i}^2 + y_i^2', savefig=True, name='circle-embedding')
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
numpyNN.plot_train_test_data(x_train, y_train, x_test, y_test, 'swirl')

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
opt_optim = Adam(mlp, learning_rate=.001)
print(mlp.summary())

# %%
spiral_embedded_logs = train_and_test_mlp(mlp, x_train_augmented, y_train, x_test_augmented, y_test, opt_loss, opt_optim, num_epoch=700)

# %%
epochs = len(spiral_embedded_logs['train_loss'])
plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
numpyNN.plot_loss(spiral_embedded_logs)
plt.subplot(1, 2, 2)
numpyNN.plot_accuracy(spiral_embedded_logs)
plt.suptitle(f'Loss-Accuracy Plots (for {epochs} epochs)', fontsize=20)
plt.tight_layout()

# %%
non_linearities =[nonlin1, nonlin2, nonlin3]
decision_boundary_plot(x_test_augmented, y_test, non_linearities, model=mlp,i=3, savefig=True,levels=1, name='spiral-embedding-db')

# %%


# %%



