import sys
import os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image

# MNIST 데이터셋 가져오기
# 처음 한 번은 몇 분 정도 걸릴 수 있다.
# load_mnist 함수는 읽은 데이터를 (훈련 이미지, 훈련 테이블), (시험 이미지, 시험 테이블) 형식으로 반환한다.

# 인수로는 normalize, flatten, one_hot_label 세 가지를 설정할 수 있고, 세 인수 모두 bool 값이다.

# normalize는 입력 이미지의 픽셀을 0.0 ~ 1.0 사이의 값으로 정규화할지를 정한다.
# False로 설정하면 입력 이미지의 픽셀은 원래 값 그대로 0 ~ 255 사이의 값을 유지한다.

# flatten은 입력 이미지를 평탄하게, 즉 1차원 배열로 만들어 줄지를 결정
# False로 설정하면 입력 이미지를 1 x 28 x 28 의 3차원 배열로,
# True로 설정하면 784(28 * 28)개의 원소로 이루어진 1차원 배열로 저장

# one_hot_label은 레이블을 원-핫 인코딩 형태로 저장할지를 결정한다.
# 원-핫 인코딩이란 정답을 뜻하는 원소만 1이고 나머지는 모두 0인 배열이다. ex) [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# False면 '7'이나 '2'와 같이 숫자 형태의 레이블을 저장하고, True일 때는 레이블을 원-핫 인코딩하여 저장한다.
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

# 각 데이터의 형상 출력
print(x_train.shape)  # (60000, 784)
print(t_train.shape)  # (60000,)
print(x_test.shape)  # (10000, 784)
print(t_test.shape)  # (10000,)


# 데이터도 확인할 겸 MNIST 이미지를 화면으로 불러오자.
# 이미지 표시에는 PIL 모듈을 사용한다.
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,) (flatten을 해줬기 때문)
img = img.reshape(28, 28)  # reshape 메소드에 원하는 형상을 인수로 지정하면 넘파이 배열의 형상을 바꿀 수 있다.
print(img.shape)  # (28, 28)

img_show(img)
