"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
from .neural_layer import NeuralLayer
from .activations import RELU,Softmax,Sigmoid,Tanh
from .objective_functions import CrossEntropy, MSE
from .optimizers import SGD

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """
    
    def __init__(self, cli_args):
        """
        Initialize the neural network.

        Args:
            cli_args: Command-line arguments for configuring the network
        """
        self.layers=[]
        self.activations=[]
        self.input_dim=784
        self.output_dim=10
        hidden_sizes=cli_args.hidden_layers
        activation=cli_args.activation
        weight_init=cli_args.weight_init
        
        #selecting activation function
        if activation == 'relu':
            activation_fn = RELU
        elif activation == "sigmoid":
            activation_fn = Sigmoid
        elif activation == "tanh":
            activation_fn = Tanh
        
        prev_dim = self.input_dim

        # Hidden layers
        for size in hidden_sizes:
            self.layers.append(NeuralLayer(prev_dim,size,weight_init))
            self.activations.append(activation_fn())
            prev_dim=size

        # Final output layer
        self.layers.append(NeuralLayer(prev_dim,self.output_dim,weight_init))
        self.activations.append(Softmax())

        # Loss
        if cli_args.loss == "cross_entropy":
            self.loss_fn = CrossEntropy()
        elif cli_args.loss == "mse":
            self.loss_fn = MSE()

        # Optimizer
        self.learning_rate = cli_args.learning_rate
        
    def forward(self, X):
        """
        Forward propagation through all layers.
        Args:
            X: Input data  
        Returns:
            Output logits
        """
        for layer, activation in zip(self.layers, self.activations):
            Z = layer.forward(X)
            X = activation.forward(Z)
        return X
    
    def backward(self):
        """
        Backward propagation to compute gradients.
        Args:
            y_true: True labels
            y_pred: Predicted outputs
        Returns:
            return grad_w, grad_b
        """
        dA = self.loss_fn.backward()
        
        for layer,activation in reversed(list(zip(self.layers, self.activations))):
            dZ = activation.backward(dA)
            dA = layer.backward(dZ)
    
    def update_weights(self):
        """
        Update weights using the optimizer.
        """
        for layer in self.layers:
            layer.W -= self.learning_rate*layer.grad_W
            layer.b -= self.learning_rate*layer.grad_b
    
    def train(self, X_train, y_train, epochs, batch_size):
        """
        Train the network for specified epochs.
        """
        num_samples = X_train.shape[0]
        for epoch in range(epochs):
            # Shuffle data
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]
            
            for i in range(0, num_samples, batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Compute loss
                loss = self.loss_fn.forward(y_batch, y_pred)
                
                # Backward pass
                self.backward()
                
                # Update weights
                self.update_weights()
                
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")
    
    def evaluate(self, X, y):
        """
        Evaluate the network on given data.
        """
        
