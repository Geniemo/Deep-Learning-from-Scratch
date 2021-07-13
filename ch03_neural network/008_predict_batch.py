import pickle
import sys
import os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


# MNIST 데이터셋을 가지고 추론을 수행하는 신경망을 구현해보자.
# 이 신경망은 입력층 뉴런을 784개, 출력층 뉴런을 10개로 구성한다.
# 입력층 뉴런이 784개인 이유는 이미지 크기가 28 * 28 = 784 이기 때문.
# 출력층 뉴런이 10개인 이유는 이 문제가 0 ~ 9 의 숫자를 구분하는 문제이기 때문.
# 은닉층은 총 두 개로, 첫 번째 은닉층에는 50개의 뉴런을, 두 번째 은닉층에는 100개의 뉴런을 배치할 것 (50과 100은 임의로 정한 값)
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


# init_network 에서는 pickle 파일인 sample_weight.pkl 에 저장된 학습된 가중치 매개변수를 읽는다.
# 이제 위의 함수들을 이용해 신경망에 의한 추론을 수행해보고 정확도도 평가해보자.
x, t = get_data()
network = init_network()

batch_size = 100  # 배치 크기
accuracy_cnt = 0
for i in range(0, len(x), batch_size):
    x_batch = x[i:i + batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)  # axis=1 로 1번째 차원을 축으로 최댓값을 찾는다.
    accuracy_cnt += np.sum(p == t[i:i + batch_size])

print('Accuracy:', float(accuracy_cnt) / len(x))
