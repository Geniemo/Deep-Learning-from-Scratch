import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

# matplotlib은 그래프를 그려주는 라이브러리이다.
# 그래프를 그리렴면 matplotlib의 pyplot 모듈을 사용한다.

# # 데이터 준비
# x = np.arange(0, 6, 0.1)  # [0, 6) 에서 0.1 간격으로 수들을 생성해 array 형태로 반환
# y1 = np.sin(x)

# # sin 그리기
# plt.plot(x, y1)
# plt.show()

# # sin과 cos 함께 그리기
# y2 = np.cos(x)
# plt.plot(x, y1, label='sin')
# plt.plot(x, y2, linestyle='--', label='cos')  # cos함수는 점선으로 그리기
# plt.xlabel('x')  # x축 이름
# plt.ylabel('y')  # y축 이름
# plt.title('sin & cos')  # 그래프 제목
# plt.legend()  # plot 메소드의 label이 범례로 사용된다.
# plt.show()

# pyplot에는 이미지를 표시해주는 메소드인 imshow()도 준비되어 있다.
# 이미지를 읽어들일때는 matplotlib.image 모듈의 imread() 메소드를 사용한다.
img = imread('ex_img.png')
plt.imshow(img)
plt.show()
