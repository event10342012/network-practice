import numpy as np
from torch import nn

nn.Sigmoid()


class Sigmoid:

    def __init__(self, x):
        self.x = x

    def __call__(self, *args, **kwargs):
        return self.forward()

    def forward(self):
        return 1 / (1 + np.exp(self.x))

    def backward(self):
        return self.forward() * (1 - self.forward())
