"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
from html import parser
import numpy as np
from ann.neural_network import NeuralNetwork
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.data_loader import load_dataset

def parse_arguments():
    """
    Parse command-line arguments for inference.
    TODO: Implement argparse with:
    - model_path: Path to saved model weights(do not give absolute path, rather provide relative path)
    - dataset: Dataset to evaluate on
    - batch_size: Batch size for inference
    - hidden_size: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='rmsprop')
    parser.add_argument('--hidden_size', nargs='+', type=int, default=[128])
    parser.add_argument('--activation', type=str, default='sigmoid')
    parser.add_argument('--loss', type=str, default='cross_entropy')
    parser.add_argument('--weight_init', type=str, default='xavier')
    parser.add_argument('--wandb_project', type=str, default='dl_assignment_1')
    parser.add_argument('--model_path', type=str, default='src/best_model.npy')
    return parser.parse_args()


def load_model(model_path):
    """
    Load trained model from disk.
    """
    weights = np.load(model_path, allow_pickle=True).item()
    return weights


def evaluate_model(model, X_test, y_test): 
    """
    Evaluate model on test data.
        
    TODO: Return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    logits = model.forward(X_test)

    predictions = np.argmax(logits, axis=1)
    true_labels = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="macro")
    precision = precision_score(true_labels, predictions, average="macro")
    recall = recall_score(true_labels, predictions, average="macro")

    loss = model.loss_fn.forward(y_test, logits)

    return {
        "logits": logits,
        "loss": loss,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


def main():
    """
    Main inference function.

    TODO: Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments()
    # load dataset
    _, _, X_test, y_test = load_dataset(args.dataset)

    # initialize model
    model = NeuralNetwork(args)

    # load weights
    weights = load_model(args.model_path)
    model.set_weights(weights)

    # evaluate
    results = evaluate_model(model, X_test, y_test)
    
    print("Evaluation complete!")
    print("Loss:",results["loss"])
    print("Accuracy:",results["accuracy"])
    print("F1 Score:",results["f1"])
    print("Precision:",results["precision"])
    print("Recall:",results["recall"])
    return results
    


if __name__ == '__main__':
    main()
