import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import numpy as np


def f(xy: tuple):
    x, y = xy
    return y ** 2 + (x * y) + np.cos(np.pi * x) ** 2


class TestData1(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        torch.random.manual_seed(0)
        self._features = 2 * torch.rand((2, 200)) - 1
        self._targets = f(self._features)

    def __len__(self):
        return self._targets.shape[0]

    def __getitem__(self, idx):
        return self._features[:, idx], self._targets[idx]

    def features(self):
        return self._features.clone()


if __name__ == "__main__":
    support = np.linspace(-1, 1, 101)
    x, y = np.meshgrid(support, support)
    z_truth = f((x, y))
    plt.pcolor(x[0, :], y[:, 0], z_truth)
    plt.colorbar()
    plt.axis('square')
    test_data = TestData1()
    test_xy = test_data.features()
    plt.plot(test_xy[0], test_xy[1], 'w.')
    plt.show()
