# %% [markdown]
# # HW1 - Exploring MLPs with PyTorch

# %% [markdown]
# 
# ## Problem 1: Simple MLP for Binary Classification
# In this problem, you will train a simple MLP to classify two handwritten digits: 0 vs 1. We provide some starter codes to do this task with steps. However, you do not need to follow the exact steps as long as you can complete the task in sections marked as <span style="color:red">[YOUR TASK]</span>.
# 
# ## Dataset Setup
# We will use the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). The `torchvision` package has supported this dataset. We can load the dataset in this way (the dataset will take up 63M of your disk space):

# %%
import torch
from torchvision import transforms, datasets
import numpy as np
import pandas as pd
import sklearn
import torch.nn as nn
from collections import defaultdict

# %%
'''
Link on how to use device
https://stackoverflow.com/questions/68820453/how-to-run-pytorch-on-macbook-pro-m1-gpu
'''

import platform, time
print(platform.mac_ver() )
print(torch.has_mps)

# if not torch.backends.mps.is_available():
#     if not torch.backends.mps.is_built():
#         print("MPS not available because the current PyTorch install was not "
#               "built with MPS enabled.")
#     else:
#         print("MPS not available because the current MacOS version is not 12.3+ "
#               "and/or you do not have an MPS-enabled device on this machine.")
    
# else:
#     device = torch.device("mps")
#     print('mps enabled')

# %%
# define the data pre-processing
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
    )

# Load the MNIST dataset 
mnist = datasets.MNIST(root='/Users/vashisth/Documents/GitHub/Intro_DL/IDL_hw1/data', 
                       train=True, 
                       download=True, 
                       transform=transform)
mnist_test = datasets.MNIST(root='/Users/vashisth/Documents/GitHub/Intro_DL/IDL_hw1/data',   # './data'
                            train=False, 
                            download=True, 
                            transform=transform)

# %%
from torch.utils.data import DataLoader, random_split

# Filter for digits 0 and 1
train_index = mnist.targets<2
mnist.data = mnist.data[train_index]
mnist.targets = mnist.targets[train_index]

test_index = mnist_test.targets<2
mnist_test.data = mnist_test.data[test_index]
mnist_test.targets = mnist_test.targets[test_index]

# %%
# Split training data into training and validation sets
train_len = int(len(mnist) *.8)
val_len = len(mnist) - train_len
train_set, val_set = random_split(mnist, [train_len, val_len])

# Define DataLoaders to access data in batches
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size = 64, shuffle=False)
test_loader = DataLoader(mnist_test, batch_size = 64, shuffle=False)

# %%
# Define your MLP
class SimpleMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(SimpleMLP, self).__init__()
        # Your code goes here
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.activation = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        # Your code goes here
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)        
        return x

hidden_dim = 5
model = SimpleMLP(in_dim=28 * 28,
                  hidden_dim=hidden_dim,
                  out_dim=2).to(device)
print(model)

# %% [markdown]
# ## Train the MLP
# To train the model, we need to define a loss function (criterion) and an optimizer. The loss function tells us how far away the model’s prediction is from the label. Once we have the loss, PyTorch can compute the gradient of the model automatically. The optimizer uses the gradient to update the model. For classification problems, we often use the Cross Entropy Loss. For the optimizer, we can use stochastic gradient descent optimizer or Adam optimizer:

# %%
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# %% [markdown]
# There are several hyper-parameters in the optimizer (please see the [PyTorch document](https://pytorch.org/docs/stable/optim.html) for details). You can play with the hyper-parameters and see how they influence the training.
# 
# Now we have almost everything to train the model. We provide a sample code to complete the training loops:

# %% [markdown]
# You can also perform validation after each epoch. But remember not to train (backward and update) on the validation dataset. Use the validation set to optimize performance. After you are done with this, report performance on the test set(You are encouraged not to use the test set for validation, i.e., use the test set only once after you are happy with the validation performance).
# 
# <span style="color:red">[YOUR TASK]</span>
# - Filter all samples representing digits "0" or "1" from the MNIST datasets. 
# - Randomly split the training data into a training set (80\% training samples) of a validation set (20% training samples).
# - Define an MLP with 1 hidden layer and train the MLP to classify the digits "0" vs "1".  Report your MLP design and training details (which optimizer, number of epochs, learning rate, etc.)
# - Keep other hyper-parameters the same, and train the model with different batch sizes: 2, 16, 128, 1024. Report the time cost, training, validation, and test set accuracy of your model
# 
# 
# In our implementations, we trained our network for 10 epochs in about 10 seconds on a laptop, getting a test accuracy of 99\% %.
# 
# One tip about the hidden layer size is to begin with a small number, say $16\sim 64$. Some people find $$\text{hidden size} = \sqrt{\text{input size}\times \text{output size}}$$ is a good choice in practice. If your model's training accuracy is too low, you can double the hidden layer size. However, if you find the training accuracy is high. Still, the validation accuracy is much lower, you may consider a smaller hidden layer size because your model has the risk of overfitting.
# 

# %%
num_epochs = 10
start_time = time.time()
for epoch in range(num_epochs):
    correct, count = 0, 0 
    for data, target in train_loader:
        # free the gradient from the previous batch
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # reshape the image into a vector
        data = data.view(data.size(0), -1)
        # model forward
        output = model(data)
        # compute the loss
        loss = criterion(output, target)
        # model backward
        loss.backward()
        # update the model paramters
        optimizer.step()
        
        # adding this for train accuracy 
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        count += data.size(0)
    
    train_acc = 100. * correct / count
    print(f'Training accuracy: {train_acc:.2f}%')

training_time = time.time()- start_time
print(training_time)

# %%
# validation set
val_loss = count = 0
correct = total = 0
for data, target in val_loader:
    data, target = data.to(device), target.to(device)
    data = data.view(data.size(0), -1)
    output = model(data)
    val_loss += criterion(output, target).item()
    count += 1
    pred = output.argmax(dim=1)
    correct += (pred == target).sum().item()
    total += data.size(0)
    
val_loss = val_loss / count
val_acc = 100. * correct / total
print(f'Validation loss: {val_loss:.2f}, accuracy: {val_acc:.2f}%')

# %%
model.eval()
correct = total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)
        output = model(data)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += data.size(0)
        
test_acc = 100. * correct / total
print(f'Test Accuracy: {test_acc:.2f}%')

# %% [markdown]
# ## Running it for different batch sizes 

# %%
'''
Two digit is a function I defined to be able to test for different batch sizes
'''
def two_digit(batch_size):
    # Define DataLoaders to access data in batches
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # Your code goes here
    val_loader = DataLoader(val_set, batch_size = batch_size, shuffle=False)
    test_loader = DataLoader(mnist_test, batch_size = batch_size, shuffle=False)
    
    model = SimpleMLP(in_dim=28 * 28,
                  hidden_dim=hidden_dim,
                  out_dim=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    num_epochs = 10
    
    # training
    start_time = time.time()
    for epoch in range(num_epochs):
        correct, count = 0, 0 
        for data, target in train_loader:
            # free the gradient from the previous batch
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            # reshape the image into a vector
            data = data.view(data.size(0), -1)
            # model forward
            output = model(data)
            # compute the loss
            loss = criterion(output, target)
            # model backward
            loss.backward()
            # update the model paramters
            optimizer.step()
            
            # adding this for train accuracy 
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            count += data.size(0)
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

        train_acc = 100. * correct / count
    training_time = time.time()- start_time
    
    # validation
    val_loss = count = 0
    correct = total = 0
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)
        output = model(data)
        val_loss += criterion(output, target).item()
        count += 1
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += data.size(0)
        
    val_loss = val_loss / count
    val_acc = 100. * correct / total
    
    # test
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += data.size(0)
            
    test_acc = 100. * correct / total
    print('Hyperopt run done')
    return training_time, train_acc, val_acc, test_acc

# %%
batch_sizes = [2, 16, 128, 1024]
results = []

for batch_size in batch_sizes:
    training_time, train_acc, val_acc, test_acc = two_digit(batch_size=batch_size)
    results.append([batch_size,training_time, train_acc, val_acc, test_acc])

# writing and saving the results to a csv
headers = ['Batch size', 'Training Time ', 'Train Acc' ,' Val Acc', 'Test Acc']
df =  pd.DataFrame(results, columns = headers)
df.to_csv('question_1.csv')
df

# %%
# to get the latex code for the table 
df = pd.read_csv('question_1.csv')
latex_table = df.to_latex(index=False)
print(latex_table)

# %% [markdown]
# # Problem 2: MNIST 10-class classification

# %% [markdown]
# Now we want to train an MLP to handle multi-class classification for all 10 digits in the MNIST dataset. We will use the full MNIST dataset without filtering for specific digits. You may modify the MLP so that it can be used for multi-class classification.
# 
# - Implement the training loop and evaluation section. Report the hyper-parameters you choose.
# - Experiment with different numbers of neurons in the hidden layer and note any changes in performance.
# - Write a brief analysis of the model's performance, including any challenges faced and how they were addressed.
# 
# In our implementations, we trained our network for 10 epochs in about 20 seconds on a laptop.
# When you define a new model, remember to update the optimizer!
# 
# 

# %%
# sigmoind activation MLP
class MulticlassMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MulticlassMLP, self).__init__()
        # Your code goes here
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.activation = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        # Your code goes here
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        
        return x

hidden_dim = int(np.sqrt(28*28*10))
model = MulticlassMLP(in_dim=28 * 28,
                  hidden_dim=hidden_dim,
                  out_dim=10).to(device)
print(model)

# %%
'''
Ten digit is a function I defined to be able to test for different hyper parameters
Input: Different hyper parameters
    - the function takes in the device (cpu vs mps) to see the difference inthe training times
    - hidden dimensions to vary the complexity of the model
    - optimizer (sgd at lr of 1e-2 and adam at lr 1e-3)
'''

def ten_digit(batch_size, hidden_dim, optimizer,  device = 'cpu'): # or mps lr=1e-3,
    device = torch.device(device)
    # Define DataLoaders to access data in batches
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size = batch_size, shuffle=False)
    test_loader = DataLoader(mnist_test, batch_size = batch_size, shuffle=False)
    
    model = MulticlassMLP(in_dim=28 * 28,
                  hidden_dim=hidden_dim,
                  out_dim=10).to(device)
    criterion = nn.CrossEntropyLoss()
    
    if optimizer == 'adam':
        lr = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        lr=1e-2
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    num_epochs = 10
    start_time = time.time()
    for epoch in range(num_epochs):
        correct, count = 0, 0 
        for data, target in train_loader:
            # free the gradient from the previous batch
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            # reshape the image into a vector
            data = data.view(data.size(0), -1)
            # model forward
            output = model(data)
            # compute the loss
            loss = criterion(output, target)
            # model backward
            loss.backward()
            # update the model paramters
            optimizer.step()
            
            # adding this for train accuracy 
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            count += data.size(0)
        
        train_acc = 100. * correct / count
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
    
    training_time = time.time()- start_time
    # print(training_time)
    
    # validation
    val_loss = count = 0
    correct = total = 0
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)
        output = model(data)
        val_loss += criterion(output, target).item()
        count += 1
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += data.size(0)
        
    val_loss = val_loss / count
    val_acc = 100. * correct / total
    
    # test
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += data.size(0)
            
    test_acc = 100. * correct / total
    print('hyperopt run done')
    return training_time, train_acc, val_acc, test_acc

# %%
# Looping over different combinations of hyper parameters
results = []
devices = ['cpu']
batch_sizes = [64, 128, 1024]
optimizers = ['adam', 'sgd']
# learning_rates= [1e-4, 1e-3, 1e-2, 1e-1]
hidden_dims = [4, 32, 64, 128]

for batch_size in batch_sizes:
    for optimizer in optimizers:
        for device in devices:
            for hidden_dim in hidden_dims:
                training_time, train_acc, val_acc, test_acc = ten_digit(batch_size=batch_size, 
                                                                        optimizer=optimizer,
                                                                        hidden_dim=hidden_dim,
                                                                        # lr = lr, 
                                                                        device=device )
                lr = 1e-3 if optimizer=='adam' else 1e-2
                print([device, batch_size, optimizer, lr, hidden_dim,  training_time, train_acc, val_acc, test_acc])
                results.append([device, batch_size, optimizer, lr, hidden_dim,  training_time, train_acc, val_acc, test_acc])

headers = ['Device', 'Batch size', 'Optimizer', 'LR', 'Hidden Dim', 
           'Training Time', 'Train Acc', 'Val Acc', 'Test Acc']
df = pd.DataFrame(results, columns=headers)
df.to_csv('sigmoid_hyperopt.csv')
df = pd.read_csv('/Users/vashisth/Documents/GitHub/Intro_DL/IDL_hw1/Question/Q2/sigmoid_hyperopt.csv')
latex_table = df.to_latex(index=False)
print(latex_table)

# %%
# ReLU activation function MLP

class MulticlassMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MulticlassMLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        # Your code goes here
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        
        return x

# %%
import pandas as pd

results = []
devices = ['cpu']
batch_sizes = [64, 128, 1024]
optimizers = ['adam', 'sgd']
# learning_rates= [1e-4, 1e-3, 1e-2, 1e-1]
hidden_dims = [4, 32, 64, 128]
for batch_size in batch_sizes:
    for optimizer in optimizers:
        for device in devices:
            for hidden_dim in hidden_dims:
                training_time, train_acc, val_acc, test_acc = ten_digit(batch_size=batch_size, 
                                                                        optimizer=optimizer,
                                                                        hidden_dim=hidden_dim,
                                                                        # lr = lr, 
                                                                        device=device )
                lr = 1e-3 if optimizer=='adam' else 1e-2
                print([device, batch_size, optimizer, lr, hidden_dim,  training_time, train_acc, val_acc, test_acc])
                results.append([device, batch_size, optimizer, lr, hidden_dim,  training_time, train_acc, val_acc, test_acc])



headers = ['Device', 'Batch size', 'Optimizer', 'LR', 'Hidden Dim', 
           'Training Time', 'Train Acc', 'Val Acc', 'Test Acc']
df = pd.DataFrame(results, columns=headers)
df.to_csv('relu_hyperopt_q2.csv')
df = pd.read_csv('/Users/vashisth/Documents/GitHub/Intro_DL/IDL_hw1/Question/Q2/relu_hyperopt_q2.csv')
latex_table = df.to_latex(index=False)
print(latex_table)

# %% [markdown]
# # Problem 3: Handling Class Imbalance in MNIST Dataset
# In this problem, we will explore how to handle class imbalance problems, which are very common in real-world applications. A modified MNIST dataset is created as follows: we choose all instances of digit “0”, and choose only 1\% instances of digit “1” for both training and test sets:

# %% [markdown]
# For such a class imbalance problem, accuracy may not be a good metric. Always predicting "0" regardless of the input can be 99\% accurate. Instead, we use the $F_1$ score as the evaluation metric:
# $$F_1 = 2\cdot\frac{\text{precision}\cdot \text{recall}}{\text{precision} + \text{recall}}$$
# where precision and recall are defined as:
# $$\text{precision}=\frac{\text{number of instances correctly predicted as "1"}}{\text{number of instances predicted as "1"}}$$
# $$\text{recall}=\frac{\text{number of instances correctly predicted as "1"}}{\text{number of instances labeled as "1"}}$$
# 
# To handle such a problem, some changes to the training may be necessary. Some suggestions include: 
# 1) Adjusting the class weights in the loss function, i.e., use a larger weight for the minority class when computing the loss.
# 2) Implementing resampling techniques (either undersampling the majority class or oversampling the minority class).
# 
# <span style="color:red">[YOUR TASK]</span>
# - Create the imbalance datasets with all "0" digits and only 1\% "1" digits.
# - Implement the training loop and evaluation section (implementing the $F_1$ metric). 
# - Ignore the class imbalance problem and train the MLP. Report your hyper-parameter details and the $F_1$ score performance on the test set (as the baseline).
# - Explore modifications to improve the performance of the class imbalance problem. Report your modifications and the $F_1$ scores performance on the test set.

# %%

import torch
from torchvision import transforms, datasets
import numpy as np
import pandas as pd
import sklearn
import torch.nn as nn
import time
from IPython.display import display
from torch.utils.data import DataLoader, random_split

# %%
device = torch.device('cpu')
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
    )

mnist = datasets.MNIST(root='/Users/vashisth/Documents/GitHub/Intro_DL/IDL_hw1/data', 
                       train=True, 
                       download=True, 
                       transform=transform)
mnist_test = datasets.MNIST(root='/Users/vashisth/Documents/GitHub/Intro_DL/IDL_hw1/data',   # './data'
                            train=False, 
                            download=True, 
                            transform=transform)

# %%
print("Frequencies: ", torch.bincount(mnist.targets))
print(len(torch.bincount(mnist.targets)))

# %%
# Define your MLP
class SimpleMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(SimpleMLP, self).__init__()
        # Your code goes here
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        # Your code goes here
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

# Your code goes here
hidden_dim = 4
model = SimpleMLP(in_dim=28 * 28,
                  hidden_dim=hidden_dim,
                  out_dim=2).to(device)
print(model)

# %%
# Your code goes here
def precision_score(labels, predictions):
    predictions, labels = np.array(labels), np.array(predictions)
    predictions_1 = np.sum(predictions==1)
    correct_1 = np.sum( (predictions==1) & (labels==1))
    precision = correct_1/ predictions_1 if predictions_1 > 0 else 1e-6
    return precision

def recall_score(labels, predictions):
    predictions, labels = np.array(labels), np.array(predictions)
    correct_1 = np.sum( (predictions==1) & (labels==1))
    labels_1 = np.sum(labels==1)
    recall = correct_1/ labels_1 if labels_1 > 0 else 1e-6
    return recall

def f1_score(labels, predictions):
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = (2 * (recall * precision)) / (precision + recall)
    return f1

# %%
# Define your MLP (again for binary classification)
class SimpleMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(SimpleMLP, self).__init__()
        # Your code goes here
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        # Your code goes here
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

# Your code goes here
hidden_dim = 4
model = SimpleMLP(in_dim=28 * 28,
                  hidden_dim=hidden_dim,
                  out_dim=2).to(device)
print(model)

# %%
'''same as before doing this so its easier to loop over hyper parameters'''

def two_digit(batch_size=64):
    model = SimpleMLP(in_dim=28 * 28,
                  hidden_dim=hidden_dim,
                  out_dim=2).to(device)
    
    # no modificaitons here 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    num_epochs = 10
    
    # training
    start_time = time.time()
    for epoch in range(num_epochs):
        correct, count = 0, 0 
        for data, target in train_loader:
            # free the gradient from the previous batch
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            # reshape the image into a vector
            data = data.view(data.size(0), -1)
            # model forward
            output = model(data)
            # compute the loss
            loss = criterion(output, target)
            # model backward
            loss.backward()
            # update the model paramters
            optimizer.step()
            
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            count += data.size(0)
        
        train_acc = 100. * correct / count

    training_time = time.time()- start_time
    
    # validation
    val_loss = count = 0
    correct = total = 0
    val_preds = []; val_labels=[]
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)
        output = model(data)
        val_loss += criterion(output, target).item()
        count += 1
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += data.size(0)
        val_preds.append(pred)
        val_labels.append(target)

    val_preds = torch.cat(val_preds).numpy()
    val_labels = torch.cat(val_labels).numpy()
    assert len(val_preds) == len(val_set)
    
    val_loss = val_loss / count
    val_acc = 100. * correct / total
    f1_validation = f1_score(labels = val_labels, predictions = val_preds)
    
    # test
    model.eval()
    correct = total = 0
    test_preds = []; test_labels=[]

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += data.size(0)
            test_preds.append(pred)
            test_labels.append(target)
        
    test_preds = torch.cat(test_preds).numpy()
    test_labels = torch.cat(test_labels).numpy()
    assert len(test_preds) == len(test_set)   
    test_acc = 100. * correct / total
    f1_test = f1_score(labels = test_labels, predictions =test_preds)
    
    return training_time, train_acc, val_acc, test_acc, f1_validation, f1_test

# %%
import random
train_0_original = [data for data in mnist if data[1] == 0]
train_1_original = [data for data in mnist if data[1] == 1]
print('Train set (before sparsing)', len(train_0_original), len(train_1_original), len(train_1_original) + len( train_0_original) )

# %% [markdown]
# <span style="color:red">[EXTRA BONUS]</span>
# 
# If the hyper-parameters are chosen properly, the baseline can perform satisfactorily on the class imbalance problem with 1% digit "1". We want to challenge the baseline and handle more class-imbalanced datasets.

# %%
headers = ['N', 'Batch size', 'Train Time ', 'Train Acc' ,' Val Acc', 'Test Acc', 'F1-Val', 'F1-Test']
question3_df =  pd.DataFrame(columns = headers)
question3_df

# %%
N_list = [100] + [250*(i+1) for i in range(8)]

# test_1 = test_1[:len(test_1) // N]  ## comment this out if you are doing sparsed vs unsparsed while runnign the code 


for N in N_list:
    train_0 = train_0_original.copy()
    train_1 =  train_1_original.copy()
    random.shuffle(train_1)
    train_1 = train_1[:len(train_1) // N]
    print(N, 'Train set (before sparsing)', len(train_0), len(train_1), len(train_1) + len( train_0) )# train_set = train_0 + train_1

    # Split training data (1s)into training and validation sets
    train_1len = int(len(train_1) *.8)
    val_1len = len(train_1) - train_1len
    train1_set, val1_set = random_split(train_1, [train_1len, val_1len])

    # Split training data (0s) into training and validation sets
    train_0len = int(len(train_0) *.8)
    val_0len = len(train_0) - train_0len
    train0_set, val0_set = random_split(train_0, [train_0len, val_0len])
    
    train_set = train0_set + train1_set
    val_set = val0_set + val1_set
    len(train_set), len(val_set)

    # creating test set
    test_0 = [data for data in mnist_test if data[1] == 0]
    test_1 = [data for data in mnist_test if data[1] == 1]
    print(N,'Test set (before sparsing)',len(test_0), len(test_1), len(test_1) + len( test_0) )

    test_1 = test_1[:len(test_1) // N]
    print(N,'Test set (after sparsing)',len(test_0), len(test_1), len(test_1) + len( test_0) )
    test_set = test_0 + test_1
    print('\n')

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size = 64, shuffle=False)
    test_loader = DataLoader(test_set, batch_size = 64, shuffle=False)
    batch_size = 64

    training_time, train_acc, val_acc, test_acc, f1_validation, f1_test = two_digit(batch_size=batch_size)
    
    row = [N, batch_size, training_time, train_acc, val_acc, test_acc, f1_validation, f1_test]
    question3_df = pd.concat([question3_df, pd.DataFrame([row], columns=headers)], ignore_index=True)

# %% [markdown]
# ## Modifications to deal with sparsity
# ### 1. Reweighting in the loss function to over upweight the sparse class 

# %%
# redefine the two_digit function to take in the weight for loss function as as parameter

def two_digit(weight, batch_size=64):
    model = SimpleMLP(in_dim=28 * 28,
                  hidden_dim=hidden_dim,
                  out_dim=2).to(device)
    
    criterion = nn.CrossEntropyLoss(weight = weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    num_epochs = 10
    
    # training
    start_time = time.time()
    for epoch in range(num_epochs):
        correct, count = 0, 0 
        for data, target in train_loader:
            # free the gradient from the previous batch
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            # reshape the image into a vector
            data = data.view(data.size(0), -1)
            # model forward
            output = model(data)
            # compute the loss
            loss = criterion(output, target)
            # model backward
            loss.backward()
            # update the model paramters
            optimizer.step()
            
            # adding this for train accuracy 
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            count += data.size(0)
        
        train_acc = 100. * correct / count
        # print(f'Training accuracy: {train_acc:.2f}%')

    training_time = time.time()- start_time
    # print(training_time)
    
    # validation
    val_loss = count = 0
    correct = total = 0
    val_preds = []; val_labels=[]
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)
        output = model(data)
        val_loss += criterion(output, target).item()
        count += 1
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += data.size(0)
        val_preds.append(pred)
        val_labels.append(target)
        # print(type(target))

    val_preds = torch.cat(val_preds).numpy()
    val_labels = torch.cat(val_labels).numpy()
    assert len(val_preds) == len(val_set)
    
    val_loss = val_loss / count
    val_acc = 100. * correct / total
    # print(f'Validation loss: {val_loss:.2f}, accuracy: {val_acc:.2f}%')
    f1_validation = f1_score(labels = val_labels, predictions = val_preds)
    # print(f'F1 score validation: {f1_validation:.2f}')
    
    # test
    model.eval()
    correct = total = 0
    test_preds = []; test_labels=[]

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += data.size(0)
            test_preds.append(pred)
            test_labels.append(target)
        
    test_preds = torch.cat(test_preds).numpy()
    test_labels = torch.cat(test_labels).numpy()
    assert len(test_preds) == len(test_set)   
    test_acc = 100. * correct / total
    # print(f'Test Accuracy: {test_acc:.2f}%')
    # print(f'Validation loss: {val_loss:.2f}, accuracy: {val_acc:.2f}%')
    f1_test = f1_score(labels = test_labels, predictions =test_preds)
    # print(f'F1 score test: {f1_test:.2f}')

    
    return training_time, train_acc, val_acc, test_acc, f1_validation, f1_test

# %%
train_0_original = [data for data in mnist if data[1] == 0]
train_1_original = [data for data in mnist if data[1] == 1]
print('Train set (before sparsing)', len(train_0_original), len(train_1_original), len(train_1_original) + len( train_0_original) )

# %%
headers = ['N', 'Batch size', 'Weight', 'Train Time ', 'Train Acc' ,' Val Acc', 'Test Acc', 'F1-Val', 'F1-Test']
question3_df =  pd.DataFrame(columns = headers)
print(question3_df)

# %%
N_list = [100] + [250*(i+1) for i in range(8)]
for N in N_list:
    train_0 = train_0_original.copy()
    train_1 =  train_1_original.copy()
    random.shuffle(train_1)
    train_1 = train_1[:len(train_1) // N]
    print(N, 'Train set (before sparsing)', len(train_0), len(train_1), len(train_1) + len( train_0) )# train_set = train_0 + train_1

    # Split training data (1s)into training and validation sets
    train_1len = int(len(train_1) *.8)
    val_1len = len(train_1) - train_1len
    train1_set, val1_set = random_split(train_1, [train_1len, val_1len])

    # Split training data (0s) into training and validation sets
    train_0len = int(len(train_0) *.8)
    val_0len = len(train_0) - train_0len
    train0_set, val0_set = random_split(train_0, [train_0len, val_0len])
    
    train_set = train0_set + train1_set
    val_set = val0_set + val1_set
    len(train_set), len(val_set)

    # creating test set
    test_0 = [data for data in mnist_test if data[1] == 0]
    test_1 = [data for data in mnist_test if data[1] == 1]
    print(N,'Test set (before sparsing)',len(test_0), len(test_1), len(test_1) + len( test_0) )

    test_1 = test_1[:len(test_1) // N]
    print(N,'Test set (after sparsing)',len(test_0), len(test_1), len(test_1) + len( test_0) )
    test_set = test_0 + test_1
    print('\n')

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size = 64, shuffle=False)
    test_loader = DataLoader(test_set, batch_size = 64, shuffle=False)

    # compensation = torch.tensor([1, N], dtype=torch.float32)
    compensation = torch.tensor([1, (train_0len/ train_1len )], dtype=torch.float32)
    weights = [[1,1], [1, int(N/10)], [1, int(N/2)], compensation]
    batch_size = 64
    results = []

    # for batch_size in batch_sizes:
    for weight in weights:
        reweight_factor = weight[1]/ weight[0]
        reweight_factor = float(reweight_factor)
        weight = torch.tensor(weight, dtype=torch.float32)
        weight = weight.to(device)
        training_time, train_acc, val_acc, test_acc, f1_validation, f1_test = two_digit(batch_size=batch_size, weight = weight)
        
        row = [N, batch_size, reweight_factor, training_time, train_acc, val_acc, test_acc, f1_validation, f1_test]
        question3_df = pd.concat([question3_df, pd.DataFrame([row], columns=headers)], ignore_index=True)

# %%
question3_df.to_csv(f'q3_hyperopt_weight_unsparsted_test.csv')
display(question3_df)

# %% [markdown]
# ## 2. Weighted Resampling in the data loader to oversample the under-represented class

# %%
def two_digit_resampling():
    model = SimpleMLP(in_dim=28 * 28,
                  hidden_dim=hidden_dim,
                  out_dim=2).to(device)
    batch_size = 64
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 10
    
    # training
    start_time = time.time()
    for epoch in range(num_epochs):
        correct, count = 0, 0 
        for data, target in train_loader:
            # free the gradient from the previous batch
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            # reshape the image into a vector
            data = data.view(data.size(0), -1)
            # model forward
            output = model(data)
            # compute the loss
            loss = criterion(output, target)
            # model backward
            loss.backward()
            # update the model paramters
            optimizer.step()
            
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            count += data.size(0)
        
        train_acc = 100. * correct / count

    training_time = time.time()- start_time
    
    # validation
    val_loss = count = 0
    correct = total = 0
    val_preds = []; val_labels=[]
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)
        output = model(data)
        val_loss += criterion(output, target).item()
        count += 1
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += data.size(0)
        val_preds.append(pred)
        val_labels.append(target)

    val_preds = torch.cat(val_preds).numpy()
    val_labels = torch.cat(val_labels).numpy()
    assert len(val_preds) == len(val_set)
    
    val_loss = val_loss / count
    val_acc = 100. * correct / total
    f1_validation = f1_score(labels = val_labels, predictions = val_preds)
    
    # test
    model.eval()
    correct = total = 0
    test_preds = []; test_labels=[]

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += data.size(0)
            test_preds.append(pred)
            test_labels.append(target)
        
    test_preds = torch.cat(test_preds).numpy()
    test_labels = torch.cat(test_labels).numpy()
    assert len(test_preds) == len(test_set)   
    test_acc = 100. * correct / total
    f1_test = f1_score(labels = test_labels, predictions =test_preds)
    
    return training_time, train_acc, val_acc, test_acc, f1_validation, f1_test

# %%
from torch.utils.data import WeightedRandomSampler
headers = ['N', 'Batch Size', 'Weight', 'Train Time ', 'Train Acc' ,' Val Acc', 'Test Acc', 'F1-Val', 'F1-Test']
question3_df_resample =  pd.DataFrame(columns = headers)

# %%
N_list = [100] + [250*(i+1) for i in range(8)]
for N in N_list:
    train_0 = train_0_original.copy()
    train_1 =  train_1_original.copy()
    random.shuffle(train_1)
    train_1 = train_1[:len(train_1) // N]
    print(N, 'Train set (before sparsing)', len(train_0), len(train_1), len(train_1) + len( train_0) )# train_set = train_0 + train_1

    # Split training data (1s)into training and validation sets
    train_1len = int(len(train_1) *.8)
    val_1len = len(train_1) - train_1len
    train1_set, val1_set = train_1[:train_1len], train_1[train_1len:]

    # Split training data (0s) into training and validation sets
    train_0len = int(len(train_0) *.8)
    val_0len = len(train_0) - train_0len
    train0_set, val0_set = train_0[:train_0len], train_0[train_0len:]
    
    # train and val set
    train_set = train0_set + train1_set
    val_set = val0_set + val1_set
    random.shuffle(train_set)
    random.shuffle(val_set)
    len(train_set), len(val_set)

    # creating test set
    test_0 = [data for data in mnist_test if data[1] == 0]
    test_1 = [data for data in mnist_test if data[1] == 1]
    print(N,'Test set (before sparsing)',len(test_0), len(test_1), len(test_1) + len( test_0) )

    test_1 = test_1[:len(test_1) // N]
    print(N,'Test set (after sparsing)',len(test_0), len(test_1), len(test_1) + len( test_0) )
    test_set = test_0 + test_1
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    print('\n')
    # compensation = torch.tensor([1, N], dtype=torch.float32)
    compensation = int(train_0len/ train_1len)
    weight_factors = [1, int(N/10), int(N/2), compensation]
    batch_size = 64
    results = []

    # for batch_size in batch_sizes:
    for weight_factor in weight_factors:
        
        weights = np.array( [1.0 if data[1] == 0 else weight_factor for data in train_set])
        weights = torch.from_numpy(weights)
        
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        
        train_loader = DataLoader(train_set, batch_size=64, sampler=sampler)
        val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
        
        training_time, train_acc, val_acc, test_acc, f1_validation, f1_test = two_digit_resampling()
        
        row = [N, batch_size, weight_factor, training_time, train_acc, val_acc, test_acc, f1_validation, f1_test]
        question3_df_resample = pd.concat([question3_df_resample, pd.DataFrame([row], columns=headers)], ignore_index=True)

# %%
question3_df_resample.to_csv(f'q3_hyperopt_resampling_unsparsed_test.csv')
display(question3_df_resample)

# %% [markdown]
# # Problem 4: Reconstruct the MNIST images by Regression
# In this problem, we want to train the MLP (with only one hidden layer) to complete a regression task: reconstruct the input image. The goal of this task is dimension reduction, and we set the hidden layer dimension to a smaller number, say 50. Once we can train the MLP to reconstruct the input images perfectly, we find an lower dimension representation of the MNIST images.
# 
# Since this is a reconstruction task, the labels of the images are not needed, and the target is the same as the inputs. Mean Squared Error (MSE) is recommended as the loss function:

# %%
device = torch.device('cpu')

# %%
# define the data pre-processing
# convert the input to the range [-1, 1].
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
    )

# Load the MNIST dataset 
# this command requires Internet to download the dataset
mnist = datasets.MNIST(root='/Users/vashisth/Documents/GitHub/Intro_DL/IDL_hw1/data', 
                       train=True, 
                       download=True, 
                       transform=transform)
mnist_test = datasets.MNIST(root='/Users/vashisth/Documents/GitHub/Intro_DL/IDL_hw1/data',   # './data'
                            train=False, 
                            download=True, 
                            transform=transform)

# %%
from torch.utils.data import DataLoader, random_split

print("Frequencies: ", torch.bincount(mnist.targets))
print(len(torch.bincount(mnist.targets)))

# %%
# Split training data into training and validation sets
train_len = int(len(mnist) *.8)
val_len = len(mnist) - train_len
train_set, val_set = random_split(mnist, [train_len, val_len])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size = 64, shuffle=False)
test_loader = DataLoader(mnist_test, batch_size = 64, shuffle=False)

# %%
# Define your MLP
class RegressionMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(RegressionMLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.activation= nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.activation_output= nn.Tanh()


    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation_output(x)
        return x

hidden_dim = 50
model = RegressionMLP(in_dim=28 * 28,
                  hidden_dim=hidden_dim,
                  out_dim=28*28).to(device)
print(model)

# %%
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# %%
num_epochs = 20
start_time = time.time()
for epoch in range(num_epochs):
    correct, count = 0, 0 
    for data, target in train_loader:
        # free the gradient from the previous batch
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # reshape the image into a vector
        data = data.view(data.size(0), -1)
        # print(data.size())
        # print(output.size())
        # model forward
        output = model(data)
        # compute the loss
        loss = criterion(output, data)
        # model backward
        loss.backward()
        # update the model paramters
        optimizer.step()
       
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

training_time = time.time()- start_time
print(training_time)

# %%
val_loss = count = 0
correct = total = 0
for data, target in val_loader:
    data, target = data.to(device), target.to(device)
    data = data.view(data.size(0), -1)
    output = model(data)
    val_loss += criterion(output, data).item()
    count += 1

val_loss = val_loss / count
print(f'Validation loss: {val_loss:.3f}')

# %%
model.eval()
loss = count= 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)
        output = model(data)
        loss += criterion(output, data).item()
        count +=1
        
test_loss= loss/count
print(f'Test Loss: {test_loss:.2f}')

# %%
# KeyError: had trouble writing over empty dic looked up that defaultdict can handle it without me having to explicitly handle constraints 
# a = {}
# if 
# a[0] = [[1,2]]
# a

# %%
check_dict = defaultdict(list)

model.eval()
with torch.no_grad():  
    for data, target in test_loader:
        data = data.view(data.size(0), -1).to(device)
        output = model(data)

        for idx, label in enumerate(target):
            label = int(label.item())
            
            #loss for each image
            individual_loss = criterion(output[idx].unsqueeze(0), data[idx].unsqueeze(0)).item()
            
            # for storing arrays and outputs
            img_np = data[idx].numpy()
            output_np = output[idx].numpy()

            check_dict[label].append([img_np, output_np, individual_loss])

# %%
'''
- https://stackoverflow.com/questions/55466298/pytorch-cant-call-numpy-on-variable-that-requires-grad-use-var-detach-num

- https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
''' 
import matplotlib.pyplot as plt

for i in range(10):
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(4,10))
    fig.suptitle('Left: Original, Right: Reconstructed')
    for j in range(5):
        axes[j,0].imshow(check_dict[i][j][0].reshape(28,28))
        axes[j,1].imshow(check_dict[i][j][1].reshape(28,28))
        axes[j,1].set_title(f'Loss {check_dict[i][j][2]:.3f}')
    fig.savefig(f'output_{i}.png')
    fig.tight_layout()


