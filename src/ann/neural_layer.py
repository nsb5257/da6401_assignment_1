"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np

class NeuralLayer:
    """
    Implements a fully connected layer with options for random and xavier weight initialisation
    Stores necessary variables for backpropagation
    """
    def __init__(self,input_dim,output_dim,weight_init="random"):
        """
        Initialize layer parameters.
        Args:
            input_dim (int): Number of input features
            output_dim (int): Number of neurons in this layer
            weight_init (str): Initialization method (random or xavier)
        """
        self.input_dim=input_dim
        self.output_dim=output_dim
        
        if weight_init=="random":
            self.W=np.random.randn(input_dim,output_dim)*0.01
        elif weight_init=="xavier":
            self.W=np.random.randn(input_dim,output_dim)*np.sqrt(1/input_dim)
            
        self.b=np.zeros((1,output_dim))
        self.grad_W=None
        self.grad_b=None
        self.input=None
        
    def forward(self,X):
        self.input=X
        Z=X@self.W+self.b
        return Z
    
    def backward(self,dZ):
        self.grad_W=self.input.T@dZ
        self.grad_b=np.sum(dZ,axis=0,keepdims=True)
        dX=dZ@self.W.T
        return dX