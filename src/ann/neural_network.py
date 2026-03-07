"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
from .neural_layer import NeuralLayer
from .activations import RELU,Sigmoid,Tanh
from .objective_functions import CrossEntropy, MSE

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """
    
    def __init__(self, cli_args):
        """
        Initialize the neural network based on CLI args
        Args:
            cli_args: Command-line arguments for configuring the network
        """
        self.layers=[]
        self.activations=[]
        self.input_dim=784
        self.output_dim=10
        hidden_sizes=cli_args.hidden_size or [10]
        activation=cli_args.activation
        weight_init=cli_args.weight_init
        #selecting activation function
        if activation == 'relu':
            activation_fn = RELU
        elif activation == "sigmoid":
            activation_fn = Sigmoid
        elif activation == "tanh":
            activation_fn = Tanh
        else:
            activation_fn = RELU
        
        prev_dim = self.input_dim
        # Hidden layers
        for size in hidden_sizes:
            self.layers.append(NeuralLayer(prev_dim,size,weight_init))
            self.activations.append(activation_fn())
            prev_dim=size
        # Final output layer
        self.layers.append(NeuralLayer(prev_dim,self.output_dim,weight_init))
        # Loss
        if cli_args.loss == "cross_entropy":
            self.loss_fn = CrossEntropy()
        elif cli_args.loss == "mse":
            self.loss_fn = MSE()
        # Optimizer
        self.optimizer = None
        
    def forward(self, X):
        """
        Forward propagation through all layers.
        Returns logits (no softmax applied)
        X is shape (b, D_in) and output is shape (b, D_out).
        b is batch size, D_in is input dimension, D_out is output dimension.
        """
        for i in range(len(self.layers) - 1):
            Z = self.layers[i].forward(X)
            X = self.activations[i].forward(Z)
        logits = self.layers[-1].forward(X)
        return logits
    
    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        Returns two numpy arrays: grad_Ws, grad_bs.
        - `grad_Ws[0]` is gradient for the last (output) layer weights,
          `grad_bs[0]` is gradient for the last layer biases, and so on.
        """
        grad_W_list = []
        grad_b_list = []

        # Backprop through layers in reverse; collect grads so that index 0 = last layer
        # gradient of loss wrt logits
        dA = self.loss_fn.backward(y_true,y_pred)

        # last layer
        dA = self.layers[-1].backward(dA)
        grad_W_list.append(self.layers[-1].grad_W)
        grad_b_list.append(self.layers[-1].grad_b)

        # hidden layers
        for i in reversed(range(len(self.activations))):
            dZ = self.activations[i].backward(dA)
            dA = self.layers[i].backward(dZ)

            grad_W_list.append(self.layers[i].grad_W)
            grad_b_list.append(self.layers[i].grad_b)
        # create explicit object arrays to avoid numpy trying to broadcast shapes
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb
        return self.grad_W, self.grad_b
    
    def update_weights(self):
        if self.optimizer is None:
            raise ValueError("Optimizer not set. Assign optimizer before training.")
        self.optimizer.update(self.layers)
    
    def train(self, X_train, y_train, epochs=1, batch_size=32):
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
                y_pred = self.forward(X_batch)
                loss = self.loss_fn.forward(y_batch, y_pred)
                self.backward(y_batch, y_pred)
                self.update_weights()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")
    
    def evaluate(self, X, y):
        y_pred = self.forward(X)
        loss = self.loss_fn.forward(y, y_pred) 
        predictions = np.argmax(y_pred, axis=1)
        if len(y.shape) > 1 and y.shape[1] > 1:
            true_labels = np.argmax(y, axis=1)
        else:
            true_labels = y
        accuracy = np.mean(predictions == true_labels)
        return loss, accuracy

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()