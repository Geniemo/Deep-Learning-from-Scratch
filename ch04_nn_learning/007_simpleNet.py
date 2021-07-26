import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # np.random.randn(m, n) -> 평균 0, 표준편차 1의 가우시안 표준정규분포 난수로 (n, m)의 매트릭스 생성

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss


net = simpleNet()
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)

t = np.array([0, 0, 1])
print(net.loss(x, t))


# numerical_gradient에 쓰려고 더미로 만든 함수
def f(W):
    return net.loss(x, t)


dW = numerical_gradient(f, net.W)
print(dW)
