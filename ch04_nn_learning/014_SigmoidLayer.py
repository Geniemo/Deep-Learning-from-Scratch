import numpy as np


# Sigmoid 계층을 구현해보자.
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        return out

    def backward(self, dout):
        dx = dout * (1 - self.out) * self.out
        return dx


