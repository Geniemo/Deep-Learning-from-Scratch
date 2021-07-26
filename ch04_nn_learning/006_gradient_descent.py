import numpy as np


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)  # x 와 형상이 같고 원소가 모두 0인 배열 생성

    for i in range(x.size):
        tmp = x[i]
        # f(x + h) 계산
        x[i] = tmp + h
        fxh1 = f(x)

        # f(x - h) 계산
        x[i] = tmp - h
        fxh2 = f(x)

        grad[i] = (fxh1 - fxh2) / (2 * h)
        x[i] = tmp

    return grad


# 경사 하강법을 직접 구현해보자.
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x


def func(x):
    return x[0] ** 2 + x[1] ** 2


init_x = np.array([-3.0, 4.0])
print(gradient_descent(func, init_x=init_x, lr=0.1, step_num=100))

# 결과값으로는 거의 0에 가까운 값이 나온다.
# 실제 최솟값도 0, 0 이므로 거의 정확한 결과를 얻은 것이다.


