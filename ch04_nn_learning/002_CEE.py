import numpy as np


# 오차제곱합 외에도 자주 사용하는 Loss function인
# 교차 엔트로피 오차, CEE(cross entropy error)도 구현해보자.
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


# 정답 (원-핫 인코딩 형식)
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# 예1: 2일 확률이 가장 높다고 추정했을 때
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print('Loss for y1:', cross_entropy_error(np.array(y1), np.array(t)))

y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print('Loss for y2:', cross_entropy_error(np.array(y2), np.array(t)))

# 위의 예에서는 y1이 더 Loss 가 적으니 정답에 가깝다고 할 수 있다.
