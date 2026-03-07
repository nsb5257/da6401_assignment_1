"""
Optimization Algorithms
Implements: SGD, Momentum, NAG, RMSProp.
"""

import numpy as np

class SGD:
    def __init__(self, learning_rate):
        self.lr=learning_rate
        
    def update(self,layers):
        for layer in layers:
            if layer.grad_W is not None:
                layer.W-=self.lr*layer.grad_W
                layer.b-=self.lr*layer.grad_b
                
class Momentum:
    def __init__(self, learning_rate, beta=0.9):
        self.lr = learning_rate
        self.beta = beta
        self.velocities = None

    def update(self, layers):
        if self.velocities is None:
            self.velocities = []
            for layer in layers:
                v_W = np.zeros_like(layer.W)
                v_b = np.zeros_like(layer.b)
                self.velocities.append((v_W, v_b))
                
        for i, layer in enumerate(layers):
            if layer.grad_W is not None:
                v_W, v_b = self.velocities[i]
                v_W = self.beta * v_W + layer.grad_W
                v_b = self.beta * v_b + layer.grad_b
                layer.W -= self.lr * v_W
                layer.b -= self.lr * v_b
                self.velocities[i] = (v_W, v_b)
                
class RMSProp:
    def __init__(self, learning_rate, beta=0.9, epsilon=1e-8):
        self.lr = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.squares = None

    def update(self, layers):
        if self.squares is None:
            self.squares = []
            for layer in layers:
                s_W = np.zeros_like(layer.W)
                s_b = np.zeros_like(layer.b)
                self.squares.append((s_W, s_b))

        for i, layer in enumerate(layers):
            if layer.grad_W is not None:
                s_W, s_b = self.squares[i]
                s_W = self.beta * s_W + (1 - self.beta) * (layer.grad_W ** 2)
                s_b = self.beta * s_b + (1 - self.beta) * (layer.grad_b ** 2)
                layer.W -= self.lr * layer.grad_W / (np.sqrt(s_W) + self.epsilon)
                layer.b -= self.lr * layer.grad_b / (np.sqrt(s_b) + self.epsilon)
                self.squares[i] = (s_W, s_b)
                
class NAG:
    def __init__(self, learning_rate, beta=0.9):
        self.lr = learning_rate
        self.beta = beta
        self.velocities = None

    def update(self, layers):

        if self.velocities is None:
            self.velocities = []
            for layer in layers:
                v_W = np.zeros_like(layer.W)
                v_b = np.zeros_like(layer.b)
                self.velocities.append((v_W, v_b))

        for i, layer in enumerate(layers):
            if layer.grad_W is not None:

                v_W, v_b = self.velocities[i]

                prev_v_W = v_W
                prev_v_b = v_b

                v_W = self.beta * v_W + layer.grad_W
                v_b = self.beta * v_b + layer.grad_b

                layer.W -= self.lr * (self.beta * prev_v_W + (1 + self.beta) * v_W)
                layer.b -= self.lr * (self.beta * prev_v_b + (1 + self.beta) * v_b)

                self.velocities[i] = (v_W, v_b)
