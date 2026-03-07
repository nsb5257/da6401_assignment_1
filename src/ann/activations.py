"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh
"""

import numpy as np

class RELU:
    def forward(self,Z):
        self.A = np.maximum(0,Z)
        return self.A

    def backward(self,dA):
        dZ=dA*(self.A>0)
        return dZ

class Sigmoid:
    def forward(self,Z):
        self.A=1/(1+np.exp(-Z))
        return self.A

    def backward(self,dA):
        dZ=dA*self.A*(1-self.A)
        return dZ
    
class Tanh:
    def forward(self,Z):
        self.A=np.tanh(Z)
        return self.A

    def backward(self,dA):
        dZ=dA*(1-self.A**2)
        return dZ