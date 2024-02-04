import numpy as np

# cite: lecture slide 20 (SGD+momentum)

class SGD:
    def __init__(self, parameters, lr_decay, decay_iter, lr=0.01, friction=0.9):
        self.parameters = parameters
        self.friction = friction
        self.lr_decay = lr_decay
        self.decay_iter = decay_iter
        self.lr = lr
        
        self.velocities = [np.zeros_like(param.data) for param in parameters]
        self.iteration = 0
        # self.velocities = np.array(self.velocities)
    
    def step(self):
        self.iteration += 1
        # lr decay every decay_iter iterations
        if self.iteration % self.n_th == 0:
            self.lr *= self.lr_decay
        
        for i, param in self.parameters:
            self.velocities[i] *= self.friction
            self.velocities[i] += param.grad
            
            param.data -= self.lr * self.velocities
            param.grad = np.zeros_like(param.data)
