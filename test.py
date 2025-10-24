import cv2
import numpy as np

img = cv2.imread(r'D:\PycharmProjects\class practice\4.jpg')
cv2.imshow('img', img)
w = np.array([[0, -1, -1, 0], [-1, 4, 5, -1], [0, -1, -1, 0]])
img_new = cv2.filter2D(img, -1, w)
cv2.imshow('img_new', img_new)
cv2.waitKey(0)
