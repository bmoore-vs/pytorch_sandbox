import os
import torch
# from torch import nn
from torch.utils.data import DataLoader

from test_data import TestData1, TestData2
import numpy as np
from matplotlib import pyplot as plt


class SimpleFFNN(torch.nn.Module):
    def __init__(self, n_input: int = 1, n_output: int = 1, n_hidden: int = 10):
        super(SimpleFFNN, self).__init__()
        self.N_INPUT = n_input
        self.N_OUTPUT = n_output
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(n_input, n_hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(n_hidden, n_output)
        )

    def forward(self, x):
        return self.layers(x)

    def grid_output(self, support):
        if not type(support) is tuple:
            support = (support,) * self.N_INPUT
        features = torch.stack([x.flatten() for x in torch.meshgrid(support, indexing='ij')])
        assert self.N_OUTPUT == 1
        out = torch.tensor([self(x) for x in features.T])
        return out.reshape([len(s) for s in support])


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    plt.figure().set_size_inches(10, 15)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    test = 2
    if test == 1:
        training_data = TestData1(1)
        testing_data = TestData1(2)
        model = SimpleFFNN(2, 1, 10)
    if test == 2:
        training_data = TestData2(1)
        testing_data = TestData2(2)
        model = SimpleFFNN(1, 1, 10)

    training_data.f_plot(show_data=True, subplot=(3, 1, 1))

    out = model.grid_output(torch.linspace(-1, 1, 201))
    training_data.f_plot(z=out, subplot=(3, 1, 2))

    training_dataloader = DataLoader(training_data, batch_size=8)
    testing_dataloader = DataLoader(testing_data, batch_size=8)

    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    epochs = 100
    for i in range(epochs):
        print(f"Epoch {i + 1}\n-------------------------------")
        train_loop(training_dataloader, model, loss_fn, optimizer)
        test_loop(testing_dataloader, model, loss_fn)

    out = model.grid_output(torch.linspace(-1, 1, 201))
    training_data.f_plot(z=out, subplot=(3, 1, 3))

    plt.show()


if __name__ == '__main__':
    main()
