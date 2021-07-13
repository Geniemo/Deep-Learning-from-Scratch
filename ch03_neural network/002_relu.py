import numpy as np
import matplotlib.pyplot as plt
# 여태 활성화 함수로 계단 함수와 시그모이드 함수를 봤는데,
# 시그모이드 함수는 신경망 분야에서 오랜 기간 사용했으나
# 최근에는 Relu(Rectified Linear Unit) 함수를 주로 이용한다.
# 이 함수는 입력이 0을 넘으면 입력을 그대로 출력하고, 0 이하이면 0을 출력하는 함수이다.


def relu(x):
    return np.maximum(0, x)  # 넘파이의 maximum 함수, 이 함수는 두 입력 중 큰 값을 선택하는 함수이다.


x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.ylim(-1, 6)
plt.show()
