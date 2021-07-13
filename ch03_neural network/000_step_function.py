import numpy as np
import matplotlib.pylab as plt
# 임계값을 경계로 출력이 바뀌는 함수를 계단 함수(step function)라고 한다.
# 활성화 함수를 계단 함수에서 다른 함수로 변경하는 것이 신경망의 세계로 나아가는 열쇠이다.
# 앞 장에서 본 퍼셉트론과 앞으로 볼 신경망의 주된 차이는 이 활성화 함수 뿐이다.
# def step_function(x):
#     if x > 0:
#         return 1
#     return 0


# 이러한 구현은 간단하지만, 인수 x는 실수만 받아들인다.
# 즉, 넘파이 배열을 인수로 넣을 수 없다는 말이다.
# 우리는 앞으로를 위해 이를 넘파이 배열도 지원하도록 바꿔보자.
def step_function(x):
    y = x > 0  # 각 원소에 대해 부등호 연산을 수행한 bool 배열이 생성
    # 원래 y는 bool배열인데, 넘파이 배열의 자료형을 변환하는 메소드인 astype(np.int)를 통해 int로 변환해주었다.
    return y.astype(np.int)


x = np.array([-1.0, 1.0, 2.0])
print(step_function(x))

# 앞서 정의한 계단 함수를 그래프로 그려보자.
x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
