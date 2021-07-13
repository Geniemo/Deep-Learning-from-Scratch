import numpy as np
import matplotlib.pylab as plt


# 이번에는 시그모이드 함수를 구현해보자
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.array([-1.0, 1.0, 2.0])
print(sigmoid(x))

# 시그모이드 함수를 그래프로 그려보자
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)  # y축 범위 지정
plt.show()
