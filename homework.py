import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('img.jpg')

# 长宽放大 2 倍
img_x2 = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

cv2.imwrite('1_x2.jpg', img_x2)
cv2.imshow('Scale x2', img_x2)
cv2.waitKey()

# 水平翻折
img_flip = cv2.flip(img, 1)

cv2.imwrite('1_1.jpg', img_flip)
cv2.imshow('Flip Horizontal', img_flip)
cv2.waitKey()
cv2.destroyAllWindows()



