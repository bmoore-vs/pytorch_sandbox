import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import numpy as np


class TestData1(Dataset):
    @staticmethod
    def f(xy: tuple):
        x, y = xy
        return y ** 2 + (x * y) + np.cos(np.pi * x) ** 2

    def __init__(self, seed=0, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        torch.random.manual_seed(seed)
        self._features = 2 * torch.rand((2, 200)) - 1
        self._targets = self.f(self._features)

    def __len__(self):
        return self._targets.shape[0]

    def __getitem__(self, idx):
        return self._features[:, idx], self._targets[idx]

    def features(self):
        return self._features.clone()

    def f_plot(self, support=np.linspace(-1, 1, 201), f=None, z=None, show_data=False, subplot=None):
        if not f:
            f = self.f
        x, y = np.meshgrid(support, support)
        if z is None:
            z = f((x, y))
        if subplot:
            plt.subplot(*subplot)
        plt.pcolor(support, support, z)
        plt.colorbar()
        plt.axis('square')
        if show_data:
            plt.plot(self._features[0], self._features[1], 'w.')
        if not subplot:
            plt.show()


def main():
    data = TestData1()
    data.f_plot(show_data=True)


if __name__ == "__main__":
    main()
