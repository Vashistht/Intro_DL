import numpy as np
# cite: lecture slide 20 (SGD+momentum)

class SGD:
    def __init__(self, lr_decay=1, decay_iter=30, lr=0.01, friction=0.9):
        self.friction = friction
        self.lr_decay = lr_decay
        self.decay_iter = decay_iter
        self.lr = lr
        # Delay the initialization of velocities
        self.v = None
    
    def initialize(self, params):
        self.params = params
        self.iteration = 0
        # Initialize velocities lazily

    def step(self):
        # Lazy initialization of velocities based on the shape of gradients
        if self.v is None:
            self.v = [np.zeros_like(param['grad']) for param in self.params if param['grad'] is not None]
        
        self.iteration += 1
        if self.iteration % self.decay_iter == 0:
            self.lr *= self.lr_decay
        
        for i, param in enumerate(self.params):
            if param['grad'] is not None:
                # Now safe to update
                self.v[i] *= self.friction
                self.v[i] += param['grad']
                param['params'] -= self.lr * self.v[i]

    def zero_grad(self):
        for param in self.params:
            if param['grad'] is not None:
                param['grad'].fill(0)

######################################
'''Taking this from my submission for HW 6, question 3 to Intro to ML (18661)
- Vashisth Tiwari HW 6'''

class Adam:
    """Adam (Adaptive Moment) optimizer.

    Parameters
    ----------
    learning_rate : float
        Learning rate multiplier.
    beta1 : float
        Momentum decay parameter.
    beta2 : float
        Variance decay parameter.
    epsilon : float
        A small constant added to the denominator for numerical stability.
    """

    def __init__(self, lr_decay=1, decay_iter=300, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
        
        self.lr_decay = lr_decay
        self.decay_iter = decay_iter
        
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.tstep = 0
        
        # Delay the initialization of m and v
        self.m = None
        self.v = None
        self.params = None
        
    def initialize(self, params):
        self.params = params


    def step(self):
        self.tstep += 1
        if self.tstep % self.decay_iter == 0:
            self.learning_rate *= self.lr_decay
       
       
        if self.m is None or self.v is None:
            # Lazy initialization of m and v based on the shape of gradients
            self.m = [np.zeros_like(param['grad']) for param in self.params if param['grad'] is not None]
            self.v = [np.zeros_like(param['grad']) for param in self.params if param['grad'] is not None]

        for i, param in enumerate(self.params):
            if param['grad'] is not None:
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param['grad']
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (param['grad'] ** 2)

                # Correct bias for m and v
                m_hat = self.m[i] / (1 - self.beta1 ** self.tstep)
                v_hat = self.v[i] / (1 - self.beta2 ** self.tstep)

                param['params'] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

                if self.tstep % self.decay_iter == 0:
                    self.learning_rate *= self.lr_decay
    
    def zero_grad(self):
        for param in self.params:
            # if param['grad'] is not None:
            param['grad'].fill(0)
