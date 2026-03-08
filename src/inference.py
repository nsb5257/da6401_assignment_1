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
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='rmsprop')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_size', nargs='+', type=int, default=[128,128,128])
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--loss', type=str, default='cross_entropy')
    parser.add_argument('--weight_init', type=str, default='xavier')
    parser.add_argument('--wandb_project', type=str, default='dl_assignment_1')
    parser.add_argument('--model_path', type=str, default='src/best_model.npy')
    parser.add_argument('--debug', action='store_true', help='Enable debug prints')
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
    weights = load_model(args.model_path)

    # ---- ADD THIS BLOCK ----
    # Infer hidden layer sizes from saved weights
    W_keys = sorted([k for k in weights.keys() if k.startswith("W")],
                    key=lambda x: int(x[1:]))

    W_shapes = [weights[k].shape for k in W_keys]

    # hidden layer sizes = output size of each layer except last
    if len(W_shapes) > 1:
        args.hidden_size = [shape[1] for shape in W_shapes[:-1]]
    else:
        args.hidden_size = []

    args.num_layers = len(args.hidden_size)

    print("Inferred hidden layers:", args.hidden_size)
    # ---- END BLOCK ----

    model = NeuralNetwork(args)
    model.set_weights(weights)

    if args.debug:
        # Print dataset shapes and types
        print("[debug] X_test shape, dtype, min/max:", X_test.shape, X_test.dtype,
              np.min(X_test), np.max(X_test))
        print("[debug] y_test shape, dtype, min/max:", y_test.shape, y_test.dtype,
              np.min(y_test), np.max(y_test))

        # If labels are one-hot, show sample argmax; otherwise, show sample ints
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            print("[debug] y_test appears one-hot. sample true labels:", np.argmax(y_test[:20], axis=1))
        else:
            print("[debug] y_test appears int labels. sample true labels:", y_test[:20])

        # Print saved weight keys & simple stats
        W_keys = sorted([k for k in weights.keys()], key=lambda x: (not x.startswith("W"), x))
        print("[debug] saved weight keys:", W_keys)
        for k in W_keys[:10]:
            v = weights[k]
            print(f"[debug] {k} shape={v.shape} norm={np.linalg.norm(v):.4f} min={v.min():.4e} max={v.max():.4e}")

        # Check the model layers shapes after init
        print("[debug] model built with layers:", len(model.layers))
        for i, layer in enumerate(model.layers):
            print(f"[debug] layer {i} W shape: {layer.W.shape}, b shape: {layer.b.shape}")

        # Run a small forward on the first 8 samples (and print logits stats)
        small_X = X_test[:8]
        logits = model.forward(small_X)
        print("[debug] small logits shape:", logits.shape)
        print("[debug] small logits sample row 0 (first 10 values):", logits[0][:10])
        preds = np.argmax(logits, axis=1)
        print("[debug] small preds:", preds)

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
