import numpy as np


def func(x):
    return np.sum(x ** 2)  # 각 원소의 제곱을 합하여 반환


# 모든 변수의 편미분을 벡터로 정리한 것을 기울기(gradient) 라고 한다.
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


print(numerical_gradient(func, np.array([3.0, 4.0])))
print(numerical_gradient(func, np.array([0.0, 2.0])))
print(numerical_gradient(func, np.array([3.0, 0.0])))
