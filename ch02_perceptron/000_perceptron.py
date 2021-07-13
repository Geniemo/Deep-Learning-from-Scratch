# 퍼셉트론은 다수의 신호를 받아 하나의 신호를 출력한다.
# 여기서 신호를 생각할 때 전류나 강물처럼 흐름이 있는 것을 상상하면 좋다.
# 전류가 전선을 타고 흐르는 전자를 내보내듯, 퍼셉트론 신호도 흐름을 만들고 정보를 앞으로 전달한다.
# 퍼셉트론 신호는 흐른다/안 흐른다(1과 0)의 두 가지 값을 가질 수 있다.
# 앞으로는 1을 신호가 흐른다, 0을 신호가 흐르지 않는다는 의미로 쓰겠다.

# 퍼셉트론을 이용해 논리회로의 게이트를 구현할 수 있다.
import numpy as np


# def AND(x1, x2):
#     w1, w2, theta = 0.5, 0.5, 0.7
#     tmp = x1 * w1 + x2 * w2
#     if tmp <= theta:
#         return 0
#     return 1


# print(AND(0, 0), AND(0, 1), AND(1, 0), AND(1, 1))  # 0 0 0 1


# bias를 도입한 AND 게이트는 다음과 같이 구현가능하다.
# bias: 뉴런이 얼마나 쉽게 활성화(결과로 1을 반환)하느냐를 조정하는 매개변수
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b  # 가중치를 각각 곱해주고 합한 후 bias를 더한다.
    if tmp <= 0:
        return 0
    return 1


print(AND(0, 0), AND(0, 1), AND(1, 0), AND(1, 1))  # 0 0 0 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(x * w) + b
    if tmp <= 0:
        return 0
    return 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(x * w) + b
    if tmp <= 0:
        return 0
    return 1


# 이와 같은 방식으로 AND, NAND, OR의 3가지 논리회로를 구현 할 수 있다.
# 그런데, XOR은 직선으로 나눌 수 없기 때문에 이와 같은 방식으로 만들 수 없다.

# 위처럼, 한 층으로 이루어진 퍼셉트론을 단층 퍼셉트론이라 하는데,
# 퍼셉트론을 쌓으면 XOR 게이트를 구현할 수 있다.
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


print(XOR(0, 0), XOR(0, 1), XOR(1, 0), XOR(1, 1))  # 0 1 1 0
