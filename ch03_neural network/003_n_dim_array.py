import numpy as np

# 아래는 지금까지 써왔던 1차원 배열이다.
a = np.array([1, 2, 3, 4])
print(a)
print(np.ndim(a))  # 배열의 차원 수는 np.ndim() 으로 확인할 수 있다.
print(a.shape)  # 배열의 형상은 인스턴스변수인 shape으로 알 수 있다.
print(a.shape[0])  # 차원에 상관없이 shape은 튜플이다.(차웡네 관계 없이 일관된 형태로 결과를 반환하기 위함)

# 아래와 같은 2차원 배열을 행렬(matrix)이라고 한다.
b = np.array([[1, 2],
              [3, 4],
              [5, 6]])
print(b)
print(np.ndim(b))
print(b.shape)

# 행렬곱은 행렬 A(i x j) 와 행렬 B(j x k) 형태의 행렬 간에 이루어지고, 그 결과로는 행렬 C(i x k)가 나온다.
# (A의 1번째 차원과 B의 0번째 차원의 원소 수가 같아야 한다. 그렇지 않으면 오류가 난다.)
# 결과물의 x행 y열의 원소는 A의 x행과 B의 y열을 곱한 결과가 된다.
# 이러한 행렬곱은 넘파이 함수 np.dot()으로 수행할 수 있다.
# np.dot() 은 입력이 1차원 배열이면 벡터를 2차원 배열이면 행렬 곱을 계산한다.
# 여기서 주의해야 할 점은 행렬의 곱에서는 교환법칙이 성립하지 않는다는 것이다.
a = np.array([[1, 2],
              [3, 4]])
b = np.array([[5, 6],
              [7, 8]])
print(np.dot(a, b))

b = np.array([5, 6])
print(np.dot(a, b))  # 형상이 (2, 2)인 행렬과 (2, ) 인 행렬의 곱
