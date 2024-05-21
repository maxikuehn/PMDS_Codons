import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score


class RNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, batch_size=1) -> None:
        """
        input_size: Number of features of your input vector
        hidden_size: Number of hidden neurons
        output_size: Number of features of your output vector
        """
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size

        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden_state) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns computed output and tanh(i2h + h2h)
        Inputs
        ------
        input: Input vector
        hidden_state: Previous hidden state
        Outputs
        -------
        out: Linear output (without activation because of how pytorch works)
        hidden_state: New hidden state matrix
        """
        input = self.i2h(input)

        hidden_state = self.h2h(hidden_state)

        hidden_state = F.tanh(input + hidden_state)
        out = self.h2o(hidden_state)
        return out, hidden_state

    def init_hidden(self) -> torch.Tensor:
        return torch.zeros(self.batch_size, self.hidden_size, requires_grad=False)


def train(model: RNN, data: DataLoader, epochs: int, optimizer: optim.Optimizer, loss_fn: nn.Module, device: torch.device = torch.device("cpu")) -> None:
    """
    Trains the model for the specified number of epochs
    Inputs
    ------
    model: RNN model to train
    data: Iterable DataLoader
    epochs: Number of epochs to train the model
    optimizer: Optimizer to use for each epoch
    loss_fn: Function to calculate loss
    """
    train_losses = {}
    model.to(device)

    model.train()
    print("=> Starting training on device:", device)

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_losses = list()

        for b, (input, label) in enumerate(data):

            hidden = model.init_hidden()

            # send tensors to device
            input, label, hidden = input.to(device), label.to(device), hidden.to(device)

            # clear gradients
            model.zero_grad()
            optimizer.zero_grad()
            loss = 0

            for i in range(input.shape[1]):
                x = input[:, i].reshape(input.shape[0], input.shape[2])
                out, hidden = model(x, hidden)

                l = loss_fn(out, label[:, i].long())
                loss += l
            # Complete gradients
            loss.backward()
            # Adjust learnable parameters
            # clip as well to avoid vanishing and exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()

            epoch_losses.append(loss.detach().item() / input.shape[1])

            if b > 0 and b % 200 == 0:
                current = b * input.shape[0]
                print_loss = np.mean(epoch_losses)
                print(f"loss: {print_loss:>7.4f}  [{current:>5d}/{len(data.dataset):>5d}]")

        train_losses[epoch] = torch.tensor(epoch_losses).mean()
        print(f'=> epoch: {epoch + 1}/{epochs}, loss: {train_losses[epoch]:.4f}, epoch time: {time.time() - epoch_start_time:.2f}s')
    print(f"=> Finished training in {time.time() - start_time:.2f}s")


def predict(model: RNN, input, device=torch.device("cpu")):
    with torch.no_grad():
        hidden = model.init_hidden()
        input, hidden = input.to(device), hidden.to(device)
        prediction = list()

        for i in range(input.shape[1]):
            x = input[:, i].reshape(input.shape[0], input.shape[2])
            out, hidden = model(x, hidden)
            out = F.softmax(out, dim=1)
            out = torch.argmax(out, dim=1)
            prediction.append(out.item())
    return np.asarray(prediction)


def evaluate(model: RNN, data: DataLoader, device=torch.device("cpu")):
    model.eval()
    model.to(device)

    predictions = np.array([])
    labels = np.array([])

    for input, label in data:
        l = np.asarray(label.view(-1))
        p = predict(model, input, device=device)

        predictions = np.append(predictions, p)
        labels = np.append(labels, l)

    _ = (predictions == labels).sum() / len(predictions)  # is same as accuracy_score
    acc = accuracy_score(labels, predictions)
    return predictions, labels, acc
