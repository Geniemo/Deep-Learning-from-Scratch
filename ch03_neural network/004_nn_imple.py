import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 출력층의 활성화 함수로 항등 함수를 이용했다.
# 출력층의 활성화 함수는 풀고자 하는 문제의 성질에 맞게 사용하면 된다.
# 예를 들어 회귀에는 항등 함수를, 2클래스 분류에는 시그모이드 함수를, 다중 클래스 분류에는 소프트맥스 함수를 쓰는 것이 일반적이다.
def identity_function(x):
    return x


# 3층 신경망을 구현해보자.
# 신경망 구현의 관례에 따라 가중치만 W와 같이 대문자로 쓰고 평향과 중간 결과 등은 모두 소문자로 썼다.
# init_network 함수는 가중치와 편향을 초기화하고 이들을 딕셔너리 변수인 network에 저장
def init_network():
    network = dict()
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network


# forward함수는 입력 신호를 출력으로 변환하는 처리 과정을 구현
def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1  # 가중치를 곱하고 편향을 더한 1층의 값
    z1 = sigmoid(a1)  # 활성화 함수를 거친 1층의 결과값
    a2 = np.dot(z1, W2) + b2  # 가중치를 곱하고 편향을 더한 2층의 값
    z2 = sigmoid(a2)  # 활성화 함수를 거친 2층의 결과값
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y


network = init_network()
x = np.array([1.0, 0.5])  # 입력
y = forward(network, x)
print(y)
