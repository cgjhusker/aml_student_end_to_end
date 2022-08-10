"""Training and evaluation."""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor

from tqdm import tqdm
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import Optimizer

from google.cloud import storage

DATA_DIR = "aml_command_artifact/data"

class NeuralNetwork(nn.Module):
    """
    Neural network that classifies Fashion MNIST-style images.
    """

    def __init__(self):
        super().__init__()
        self.sequence = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 20),
                                      nn.ReLU(), nn.Linear(20, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_prime = self.sequence(x)
        return y_prime

def fit(device: str, dataloader: DataLoader, model: nn.Module,
        loss_fn: CrossEntropyLoss, optimizer: Optimizer) -> Tuple[float, float]:
    """
    Trains the given model for a single epoch.
    """
    loss_sum = 0
    correct_item_count = 0
    item_count = 0

    model.to(device)
    model.train()

    for (x, y) in tqdm(dataloader):
        x = x.float().to(device)
        y = y.long().to(device)

        (y_prime, loss) = _fit_one_batch(x, y, model, loss_fn, optimizer)

        correct_item_count += (y_prime.argmax(1) == y).sum().item()
        loss_sum += loss.item()
        item_count += len(x)

    average_loss = loss_sum / item_count
    accuracy = correct_item_count / item_count

    return (average_loss, accuracy)


def _fit_one_batch(x: torch.Tensor, y: torch.Tensor, model: NeuralNetwork,
                   loss_fn: CrossEntropyLoss,
                   optimizer: Optimizer) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Trains a single minibatch (backpropagation algorithm).
    """
    y_prime = model(x)
    loss = loss_fn(y_prime, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return (y_prime, loss)


def evaluate(device: str, dataloader: DataLoader, model: nn.Module,
             loss_fn: CrossEntropyLoss) -> Tuple[float, float]:
    """
    Evaluates the given model for the whole dataset once.
    """
    loss_sum = 0
    correct_item_count = 0
    item_count = 0

    model.to(device)
    model.eval()

    with torch.no_grad():
        for (x, y) in dataloader:
            x = x.float().to(device)
            y = y.long().to(device)

            (y_prime, loss) = _evaluate_one_batch(x, y, model, loss_fn)

            correct_item_count += (y_prime.argmax(1) == y).sum().item()
            loss_sum += loss.item()
            item_count += len(x)

        average_loss = loss_sum / item_count
        accuracy = correct_item_count / item_count

    return (average_loss, accuracy)


def _evaluate_one_batch(
        x: torch.tensor, y: torch.tensor, model: NeuralNetwork,
        loss_fn: CrossEntropyLoss) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluates a single minibatch.
    """
    with torch.no_grad():
        y_prime = model(x)
        loss = loss_fn(y_prime, y)

    return (y_prime, loss)

def load_train_val_data(
        data_dir: str, batch_size: int,
        training_fraction: float) -> Tuple[DataLoader, DataLoader]:
    """
    Returns two DataLoader objects that wrap training and validation data.
    Training and validation data are extracted from the full original training
    data, split according to training_fraction.
    """
    full_train_data = datasets.FashionMNIST(data_dir,
                                            train=True,
                                            download=True,
                                            transform=ToTensor())
    full_train_len = len(full_train_data)
    train_len = int(full_train_len * training_fraction)
    val_len = full_train_len - train_len
    (train_data, val_data) = random_split(dataset=full_train_data,
                                          lengths=[train_len, val_len])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    return (train_loader, val_loader)


def save_model(model: nn.Module) -> None:
    """
    Saves the trained model.
    """
    code_files = ["neural_network.py", "utils_train_nn.py"]
    code_paths = [
        Path(Path(__file__).parent, code_file) for code_file in code_files
    ]

    # Logs the model as an artifact.
    model_info = mlflow.pytorch.log_model(pytorch_model=model,
                                          artifact_path="model_artifact",
                                          code_paths=code_paths)

    logging.info("model_uri=%s", model_info.model_uri)


def train(data_dir: str, device: str) -> None:
    """
    Trains the model for a number of epochs, and saves it.
    """
    learning_rate = 0.1
    batch_size = 64
    epochs = 5

    (train_dataloader,
     val_dataloader) = load_train_val_data(data_dir, batch_size, 0.8)
    model = NeuralNetwork()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        logging.info("Epoch %d", epoch + 1)
        (training_loss, training_accuracy) = fit(device, train_dataloader,
                                                 model, loss_fn, optimizer)
        (validation_loss,
         validation_accuracy) = evaluate(device, val_dataloader, model, loss_fn)

        metrics = {
            "training_loss": training_loss,
            "training_accuracy": training_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy
        }
        mlflow.log_metrics(metrics, step=epoch)

    save_model(model)


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", default=DATA_DIR)
    args = parser.parse_args()
    logging.info("input parameters: %s", vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train(**vars(args), device=device)


if __name__ == "__main__":
    main()
