import os
import torch
#from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from test_data_1 import TestData1


class SimpleFFNN(torch.nn.Module):
    def __int__(self):
        super(self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(2, 10),
            torch.nn.Tanh(),
            torch.nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.layers(x)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

training_data = TestData1(1)
test_data = TestData1(2)




