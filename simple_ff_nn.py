import os
import torch
# from torch import nn
from torch.utils.data import DataLoader

from test_data import TestData1
import numpy as np


class SimpleFFNN(torch.nn.Module):
    def __init__(self, n_input: int = 1, n_output: int = 1):
        super(SimpleFFNN, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(n_input, 10),
            torch.nn.Tanh(),
            torch.nn.Linear(10, n_output)
        )

    def forward(self, x):
        return self.layers(x)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

training_data = TestData1(1)
test_data = TestData1(2)

network = SimpleFFNN(2, 1)

training_data.f_plot(show_data=True)

support = np.linspace(-1, 1, 101)
x, y = np.meshgrid(support, support)
# xy = torch.from_numpy(np.vstack((x.flatten(), y.flatten())))
z = [network.forward(torch.tensor((xx, yy))) for xx, yy in zip(x.flatten(), y.flatten())]
