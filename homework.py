import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('1.jpg')
M = np.array([[1,0,100],[0,1,200]])
img1_t = cv2.warpAffine(img1, M, (img1.shape[1], img1.shape[0]))

