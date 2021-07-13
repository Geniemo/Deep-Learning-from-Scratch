import numpy as np


# 소프트맥스 함수를 구현해보자
# def softmax(a):
#     exp_a = np.exp(a)
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a / sum_exp_a
#     return y


# 위에서 구현한 softmax 함수는 softmax 함수의 식을 제대로 표현하고 있지만,
# 컴퓨터로 계산할 때에는 overflow 문제로 인해 결함이 있을 수 있다.
# 따라서, 입력 신호 중 최댓값을 이용해 소프트맥스의 overflow 문제를 개선할 수 있다.
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


a = np.array([1010, 1000, 900])
# print(np.exp(a) / np.sum(np.exp(a)))  # 개선되지 않은 softmax, 결과는 [nan, nan, nan] 으로 계산이 제대로 되지 않는다.
y = softmax(a)
print(y)  # overflow 문제를 개선한 softmax 함수로는 계산이 제대로 이루어진다.
print(np.sum(y))  # softmax 함수의 출력은 0에서 1.0 사이의 실수이고, softmax 함수의 출력의 총합은 1이다.
# 이러한 성질때문에 softmax 함수의 출력을 확률로 해석가능하다.

# 가령, 다음과 같은 입력이 있다고 할 때,
a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)  # [0.01821127 0.24519181 0.73659691] 과 같은 출력은
# 약 1% 확률로 0번째 클래스, 25% 확률로 1번째 클래스, 74% 확률로 2번째 클래스 와 같이 해석할 수 있다.

# 여기서 주의점은 소프트맥스 함수를 적용해도 각 원소의 대소 관계는 변하지 않는다는 것이다. (exp가 단조 증가 함수이기 때문)
# 신경망을 이용한 분류에서는 일반적으로 가장 큰 출력을 내는 뉴런에 해당하는 클래스로만 인식한다.
# 그리고 소프트맥스 함수를 적용해도 출력이 가장 큰 뉴런의 위치는 달라지지 않는다.
# 결과적으로 신경망으로 분류할 때는 출력층의 소프트맥스 함수를 생략해도 된다.
# 현업에서도 지수 함수 계산에 드는 자원 낭비를 줄이고자 출력층의 소프트맥스 함수는 생략하는 것이 일반적이다.
