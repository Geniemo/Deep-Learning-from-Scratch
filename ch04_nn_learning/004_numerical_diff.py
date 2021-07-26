import numpy as np
import matplotlib.pylab as plt


# 미분을 수식으로 표현해보자면, 다음과 같이 표현할 수 있다.
def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)  # 중심 차분 혹은 중앙 차분


# 예시를 하나 만들어 테스트해보자.
def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x


# 위에서 만든 함수를 그래프로 만들어보자.
x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(x, y)
plt.show()

# x가 5일 때와 10일 때 이 함수의 미분을 구해보자.
print(numerical_diff(function_1, 5))  # 0.1999999999990898, 실제 값이 0.2
print(numerical_diff(function_1, 10))  # 0.2999999999986347, 실제 값이 0.3 이므로 거의 같은 값이라고 봐도 될 정도의 오차이다.
