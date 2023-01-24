import os
import torch
# from torch import nn
from torch.utils.data import DataLoader

from test_data import TestData1
import numpy as np
from matplotlib import pyplot as plt


class SimpleFFNN(torch.nn.Module):
    def __init__(self, n_input: int = 1, n_output: int = 1):
        super(SimpleFFNN, self).__init__()
        self.N_INPUT = n_input
        self.N_OUTPUT = n_output
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(n_input, 10),
            torch.nn.Tanh(),
            torch.nn.Linear(10, n_output)
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


def main():
    plt.figure().set_size_inches(10, 15)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    training_data = TestData1(1)
    test_data = TestData1(2)

    network = SimpleFFNN(2, 1)

    training_data.f_plot(show_data=True, subplot=(3, 1, 1))

    out = network.grid_output(torch.linspace(-1, 1, 201))
    training_data.f_plot(z=out, subplot=(3, 1, 2))

    plt.show()


if __name__ == '__main__':
    main()
