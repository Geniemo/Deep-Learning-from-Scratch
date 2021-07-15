import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)  # (60000, 784)
print(t_train.shape)  # (60000, 10)

# 아래의 코드를 통해 train_size 개에서 batch_size 개를 무작위로 골라낼 수 있다.
train_size = x_train.shape[0]
batch_size = 10
# np.random.choice(전체 개수, 고를 개수)
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

# 미니배치 같은 배치 데이터를 지원하는 교차 엔트로피 오차를 구하는 함수를 구현해보자.
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


# 정답 레이블이 원-핫 인코딩이 아니라 '2', '7' 등의 숫자 레이블로 주어졌을 때의
# 교차 엔트로피는 다음과 같이 구현할 수 있다.
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
