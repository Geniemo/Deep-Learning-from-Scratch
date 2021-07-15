import numpy as np


# 가장 많이 쓰이는 손실 함수인
# 오차제곱합, SSE(sum_squares_error)를 직접 구현해보자.
def sum_squares_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


# 정답 (원-핫 인코딩 형식)
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# 예1: 2일 확률이 가장 높다고 추정했을 때
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print('Loss for y1:', sum_squares_error(np.array(y1), np.array(t)))

y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print('Loss for y2:', sum_squares_error(np.array(y2), np.array(t)))

# 위의 예에서는 y1이 더 Loss 가 적으니 정답에 가깝다고 할 수 있다.
