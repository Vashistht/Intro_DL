import numpy as np

'''
Note: Most of the code is from what I wrote / was given for HW 6 Intro to ML (18661).
I modified it to fit the requirements here
'''

# function for momentum is from slides 21 from lecture on bags of tricks slides 


class SGD:
    def __init__(self, model, lr=0.01, momentum=1, lr_decay=1, decay_iter=100):
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.lr_decay = lr_decay
        self.decay_iter = decay_iter
        self.iteration = 0
        self.v_W = [np.zeros_like(layer.W) for layer in model.layers if hasattr(layer, 'W')] # get weights.shape
        self.v_b = [np.zeros_like(layer.b) for layer in model.layers if hasattr(layer, 'b')] # get biases.shape

    def step(self):
        self.iteration += 1
        if self.iteration % self.decay_iter == 0:
            self.lr *= self.lr_decay
            
        for i, layer in enumerate([l for l in self.model.layers if hasattr(l, 'W')]):
            # Apply updates with momentum
            self.v_W[i] = self.momentum * self.v_W[i] + layer.dLdW
            layer.W -= self.lr * self.v_W[i]

            self.v_b[i] = self.momentum * self.v_b[i] + layer.dLdb
            layer.b -= self.lr * self.v_b[i]

    def zero_grad(self):
        for layer in self.model.layers:
            if hasattr(layer, 'dLdW'):
                layer.dLdW.fill(0.0)
            if hasattr(layer, 'dLdb'):
                layer.dLdb.fill(0.0)


###
'''Took from IML HW 6 and modified it to fit the requirements here'''

class Adam:
    def __init__(self, model, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7, lr_decay=1, decay_iter=100):
        self.model = model
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.lr_decay = lr_decay
        self.decay_iter = decay_iter
        self.iteration = 0
        # initialize moments
        self.m_W = [np.zeros_like(layer.W) for layer in model.layers if hasattr(layer, 'W')]
        self.v_W = [np.zeros_like(layer.W) for layer in model.layers if hasattr(layer, 'W')]
        self.m_b = [np.zeros_like(layer.b) for layer in model.layers if hasattr(layer, 'b')]
        self.v_b = [np.zeros_like(layer.b) for layer in model.layers if hasattr(layer, 'b')]

    def step(self):
        self.iteration += 1
        if self.iteration % self.decay_iter == 0:
            self.learning_rate *= self.lr_decay
            
        for i, layer in enumerate([l for l in self.model.layers if hasattr(l, 'W')]):
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * layer.dLdW
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * (layer.dLdW ** 2)
            m_hat_W = self.m_W[i] / (1 - self.beta1 ** self.iteration)
            v_hat_W = self.v_W[i] / (1 - self.beta2 ** self.iteration)
            layer.W -= self.learning_rate * m_hat_W / (np.sqrt(v_hat_W) + self.epsilon)
            
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * layer.dLdb
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (layer.dLdb ** 2)
            m_hat_b = self.m_b[i] / (1 - self.beta1 ** self.iteration)
            v_hat_b = self.v_b[i] / (1 - self.beta2 ** self.iteration)
            layer.b -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
    
    def zero_grad(self):
            for layer in self.model.layers:
                if hasattr(layer, 'dLdW'):
                    layer.dLdW.fill(0.0)
                if hasattr(layer, 'dLdb'):
                    layer.dLdb.fill(0.0)