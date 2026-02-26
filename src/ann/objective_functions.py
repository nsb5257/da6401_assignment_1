"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np

class CrossEntropy:
    """
    Cross-Entropy loss for multi-class classification
    When inputs are y_true (one-hot encoded labels) and y_pred (softmax probabilities)
    """
    def forward(self,y_true,y_pred):
        "Compute average cross-entropy loss over batch"
        self.y_true=y_true
        self.y_pred=y_pred
        eps=1e-9 #to avoid log(0)
        loss=-np.sum(y_true*np.log(y_pred+eps))/y_true.shape[0]
        return loss
    
    def backward(self):
        "Compute gradient of CrossEntropy loss w.r.t logits"
        return (self.y_pred-self.y_true)/self.y_true.shape[0]
    
class MSE:
    """
    Computes average squared difference between predictions and targets.
    """
    def forward(self,y_true,y_pred):
        "Compute average MSE loss over batch"
        self.y_true=y_true
        self.y_pred=y_pred
        loss=np.mean((y_true-y_pred)**2)
        return loss
    
    def backward(self):
        "Compute gradient of MSE loss w.r.t predictions"
        batch_size = self.y_true.shape[0]
        return 2*(self.y_pred-self.y_true)/batch_size