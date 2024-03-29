import random
import numpy as np
import matplotlib.pyplot as plt

def linearData(n_sample=400):
	theta = np.random.rand() * 2 * np.pi
	w_star = np.array([[np.cos(theta), np.sin(theta)]])
	margin = 0.1
	noise = 0.1
	#  create data
	X = 2 * np.random.rand(n_sample, 2) - 1
	label = (X @ w_star.T) > 0
	label = label.astype(float)
	# create margin
	idx = (label * (X @ w_star.T)) < margin
	X = X + margin * ((idx * label) @ w_star)
	# add noise
	noise_x = noise * (2 * np.random.rand(n_sample, 2) - 1)
	X = X + noise_x
	return X, label


def XORData(n_sample=400):
	margin = 0.1
	noise = 0.1
	# create data
	X = 2 * np.random.rand(n_sample, 2) - 1
	label = (X[:, 0] * X[:, 1]) > 0
	label = label.astype(float).reshape((-1, 1))
	# create margin
	pos_flag = X >= 0
	X = X + 0.5 * margin * pos_flag
	X = X - 0.5 * margin * (~pos_flag)
	# add noise
	noise_x = noise * (2 * np.random.rand(n_sample, 2) - 1)
	X = X + noise_x
	return X, label


def circleData(n_sample=400):
	noise = 0.05
	# create data
	X = 2 * np.random.rand(n_sample, 2) - 1
	dist = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
	label = dist <= 0.5
	label = label.astype(float).reshape((-1, 1))
	# add noise
	noise_x = noise * (2 * np.random.rand(n_sample, 2) - 1)
	X = X + noise_x
	return X, label


def sinusoidData(n_sample=400):
	noise = 0.05
	# create data
	X = 2 * np.random.rand(n_sample, 2) - 1
	label = (np.sin(np.sum(X, axis=- 1) * 2 * np.pi) > 0)
	label = label.astype(float).reshape((-1, 1))
	# add noise
	noise_x = noise * (2 * np.random.rand(n_sample, 2) - 1)
	X = X + noise_x
	return X, label


def swissrollData(n_sample=400):
	noise = 0.05
	nHalf = int(n_sample / 2)
	# create data
	t = np.random.rand(nHalf, 1)
	x1 = t * np.cos(2 * np.pi * t * 2)
	y1 = t * np.sin(2 * np.pi * t * 2)
	t = np.random.rand(n_sample - nHalf, 1)
	x2 = (-t) * np.cos(2 * np.pi * t * 2)
	y2 = (-t) * np.sin(2 * np.pi * t * 2)
	xy1 = np.concatenate([x1, y1], axis=1)
	xy2 = np.concatenate([x2, y2], axis=1)
	X = np.concatenate([xy1, xy2], axis=0)
	label = np.concatenate([np.ones((nHalf, 1)), np.zeros((n_sample - nHalf, 1))], axis=0)
	# add noise
	noise_x = noise * (2 * np.random.rand(n_sample, 2) - 1)
	X = X + noise_x
	return X, label


def sample_data(data_name='circle', nTrain=200, nTest=200, random_seed=0,):
	"""
	Data generation function
	:param data_name: linear-separable, XOR, circle, sinusoid, swiss-roll
	:return:
	"""
	random.seed(random_seed)
	np.random.seed(random_seed)
	n_sample = nTrain + nTest
	if data_name == 'linear-separable':
		X, label = linearData(n_sample)
	elif data_name == 'XOR':
		X, label = XORData(n_sample)
	elif data_name == 'circle':
		X, label = circleData(n_sample)
	elif data_name == 'sinusoid':
		X, label = sinusoidData(n_sample)
	elif data_name == 'swiss-roll':
		X, label = swissrollData(n_sample)
	else:
		raise NotImplementedError


	indices = np.random.permutation(n_sample)
	train_idx, test_idx = indices[:nTrain], indices[nTrain:]
	x_train = X[train_idx]
	y_train = label[train_idx]
	x_test = X[test_idx]
	y_test = label[test_idx]
	return x_train, y_train, x_test, y_test


def plot_loss(logs,savefig = False, figname = None):
	"""
	Function to plot training and validation/test loss curves
	:param logs: dict with keys 'train_loss','test_loss' and 'epochs', where train_loss and test_loss are lists with 
				the training and test/validation loss for each epoch
	"""
	# plt.subplot(1, 2, 1)
	t = np.arange(len(logs['train_loss']))
	plt.plot(t, logs['train_loss'], label='train loss', lw=3)
	plt.plot(t, logs['test_loss'], label='test loss', lw=3)
	min_train_loss = min(logs['train_loss'])
	min_test_loss = min(logs['test_loss'])
	
	plt.title(f"Loss Curves (Min Train Loss: {min_train_loss:.4f}, Min Test Loss: {min_test_loss:.4f})", fontsize = 16)
	plt.grid(1)
	plt.xlabel('Epochs',fontsize=15)
	plt.ylabel('Loss',fontsize=15)
	plt.legend(fontsize=15)
	if savefig:
		directory = '/Users/vashisth/Documents/GitHub/Intro_DL/IDL_hw2/source/plots/'
		figname = directory+ figname + '.png'
		plt.savefig(figname, dpi=300)



def plot_accuracy(logs):
	"""
	Function to plot training and validation/test loss curves
	:param logs: dict with keys 'train_accuracy','test_accuracy' and 'epochs', where train_accuracy and test_accuracy are lists with 
				the training and test/validation loss for each epoch
	"""
	# plt.subplot(1, 2, 1)
	t = np.arange(len(logs['train_accuracy']))
	plt.plot(t, logs['train_accuracy'], label='train_accuracy', lw=3)
	plt.plot(t, logs['test_accuracy'], label='test_accuracy', lw=3)
 
	max_train_accuracy = max(logs['train_accuracy'])
	max_test_accuracy = logs['test_accuracy'][np.argmax(logs['train_accuracy'])]
	plt.title(f"Accuracy Curves (Max Train Acc.: {max_train_accuracy:.4f}, Test Acc.: {max_test_accuracy:.4f})", fontsize= 16)
	plt.grid(1)
	plt.xlabel('Epochs',fontsize=15)
	plt.ylabel('Accuracy',fontsize=15)
	plt.legend(fontsize=15)




def plot_decision_boundary(X, y, model, boundry_level=None):
    """
    Plots the decision boundary for the model prediction
    :param X: input data
    :param y: true labels
    :param model: your MLP model
    :boundry_level: Determines the number and positions of the contour lines / regions.
    :return:
    # """
    # print(X.shape)
    # print(y.shape)
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    def predict(X): # added this to work with my mlp
        y_pred = model.forward(X)
        return np.argmax(y_pred, axis=1)


    # Use the predict function as pred_fn
    Z = predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.7, levels=boundry_level, cmap='viridis_r')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.scatter(X[:, 0], X[:, 1], c=y.reshape(-1), alpha=0.7,s=50, cmap='viridis_r', edgecolor='k')


def plot_train_test_data(x_train, y_train, x_test, y_test, figname):
    # Flatten y_train to a 1D array for boolean indexing
	y_train_flat = y_train.flatten()
	y_test_flat = y_test.flatten()
 
	# Now, separate x_train data points
	x_train_0 = x_train[y_train_flat == 0]
	x_train_1 = x_train[y_train_flat == 1]
	x_test_0 = x_test[y_test_flat == 0]
	x_test_1 = x_test[y_test_flat == 1]
 
	plt.figure(figsize=(20, 8))
	plt.subplot(1, 2, 1)
	plt.scatter(x_train_0[:, 0], x_train_0[:, 1], c='red', label='0')
	plt.scatter(x_train_1[:, 0], x_train_1[:, 1], c='blue', label=' 1')
	plt.xlabel('Feature 1')
	plt.ylabel('Feature 2')
	plt.title('Training Data')
	plt.legend()
	plt.subplot(1, 2, 2)
	plt.scatter(x_test_0[:, 0], x_test_0[:, 1], c='red', label='0')
	plt.scatter(x_test_1[:, 0], x_test_1[:, 1], c='blue', label=' 1')
	plt.xlabel('Feature 1')
	plt.ylabel('Feature 2')
	plt.title('Test Data')
	plt.suptitle('Train-Test Data', fontsize=20)
	plt.tight_layout()
	plt.legend()
	directory = '/Users/vashisth/Documents/GitHub/Intro_DL/IDL_hw2/source/plots/'
	figname = directory+ 'train_test_'+ figname + '.png'
	plt.savefig(figname, dpi=300)
	plt.show()


def plot_stats (logs, model, x_train, y_train, x_test, y_test, fig1_name, fig2_name):
	directory = '/Users/vashisth/Documents/GitHub/Intro_DL/IDL_hw2/source/plots/'
	fig1_name = directory+ fig1_name + 'loss_acc.png'
	fig2_name = directory+ fig2_name + 'boundary.png'
	epochs = len(logs['train_loss'])
	plt.figure(figsize=(20, 8))
	plt.subplot(1, 2, 1)
	# plt.title("Loss Curves")
	plot_loss(logs)
	# test set
	plt.subplot(1, 2, 2)
	# plt.title("Accuracy Curves")
	plot_accuracy(logs)
	plt.suptitle(f'Loss-Accuracy Plots (for {epochs} epochs)', fontsize=20)
	plt.tight_layout()
	plt.savefig(fig1_name, dpi=300)
	# train set



	plt.figure(figsize=(20, 8))
	plt.subplot(1, 2, 1)
	plt.title("Train Set Decision Boundary", fontsize = 16)
	plot_decision_boundary(x_train, y_train, model , boundry_level=None)
	# test set
	plt.subplot(1, 2, 2)
	plt.title("Test Set Decision Boundary",fontsize = 16)
	plot_decision_boundary(x_test, y_test, model , boundry_level=None)
	plt.suptitle('Decision Boundary Plots', fontsize=20)
	plt.tight_layout()
	plt.savefig(fig2_name, dpi=300)