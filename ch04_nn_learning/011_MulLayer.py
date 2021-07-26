# 계산 그래프의 곱셈 노드를 구현해보자.
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    # 상류에서 넘어온 미분(dout)에 순전파 때의 값을 서로 바꿔서 곱한 후 하류로 흘린다.
    def backward(self, dout):
        dx = dout * self.y  # x와 y를 바꾸기
        dy = dout * self.x

        return dx, dy


# 책의 그림 5-16의 순전파를 다음과 같이 구현할 수 있다.
apple = 100
apple_num = 2
tax = 1.1

# 계층들
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# 순전파
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print(price)

# 각 변수에 대한 미분은 backward() 에서 구할 수 있다.
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(dapple, dapple_num, dtax)
