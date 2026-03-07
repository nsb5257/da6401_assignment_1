"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import numpy as np
import wandb
import json

from ann.neural_network import NeuralNetwork
from ann.optimizers import SGD, Momentum, NAG, RMSProp
from utils.data_loader import load_dataset

def parse_arguments():
    """
    Parse command-line arguments.
    
    TODO: Implement argparse with the following arguments:
    - dataset: 'mnist' or 'fashion_mnist'
    - epochs: Number of training epochs
    - batch_size: Mini-batch size
    - learning_rate: Learning rate for optimizer
    - optimizer: 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'
    - hidden_size: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    - loss: Loss function ('cross_entropy', 'mse')
    - weight_init: Weight initialization method
    - wandb_project: W&B project name
    - model_save_path: Path to save trained model (do not give absolute path, rather provide relative path)
    """
    parser = argparse.ArgumentParser(description='Train a neural network')
    parser.add_argument('--dataset',type=str)
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--batch_size',type=int)
    parser.add_argument('--learning_rate',type=float)
    parser.add_argument('--optimizer',type=str)
    parser.add_argument('--hidden_size',nargs='+',type=int,default=[128, 128])
    parser.add_argument('--activation',type=str)
    parser.add_argument('--loss',type=str)
    parser.add_argument('--weight_init',type=str)
    parser.add_argument('--wandb_project',type=str, default='dl_assignment_1')
    parser.add_argument('--model_save_path',type=str, default='best_model.npy')
    
    return parser.parse_args()


def main():
    """
    Main training function.
    """
    args = parse_arguments()

    # Initialize W&B
    wandb.init(project=args.wandb_project, config=vars(args))

    # Load dataset
    X_train, y_train, X_test, y_test = load_dataset(args.dataset)
    
    #Log sample images
    table = wandb.Table(columns=["image", "label"])
    for class_id in range(10):
        indices = np.where(np.argmax(y_train, axis=1) == class_id)[0][:5]
        for idx in indices:
            image = X_train[idx].reshape(28,28)
            table.add_data(wandb.Image(image),class_id)
            
    wandb.log({"sample_images": table})

    # Create model
    model = NeuralNetwork(args)

    # Select optimizer
    if args.optimizer == "sgd":
        model.optimizer = SGD(args.learning_rate)
    elif args.optimizer == "momentum":
        model.optimizer = Momentum(args.learning_rate)
    elif args.optimizer == "nag":
        model.optimizer = NAG(args.learning_rate)
    elif args.optimizer == "rmsprop":
        model.optimizer = RMSProp(args.learning_rate)

    model.train(X_train, y_train, args.epochs, args.batch_size)
    
    loss, accuracy = model.evaluate(X_test, y_test)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)

    wandb.log({"test_loss": loss, "test_accuracy": accuracy})

    # Save model
    best_weights = model.get_weights()
    np.save(args.model_save_path, best_weights)
    with open('best_config.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    print("Training complete!")

if __name__ == '__main__':
    main()
