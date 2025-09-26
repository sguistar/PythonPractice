import cv2
import numpy as np
import matplotlib.pyplot as plt

JAX_ENABLE_X64 = True
img = cv2.imread(r'D:\\PycharmProjects\\class practice\\test.jpg')
H,W,C = img.shape
print(H,W,C)
pst1 = np.float32([[50,50],[200,50],[50,200]])
pst2 = np.float32([[10,100],[200,50],[100,250]])

M = cv2.getAffineTransform(pst1,pst2)
print(M)
affine_img = cv2.warpAffine(img,M,(W,H))
cv2.imshow('affine_img',affine_img)
cv2.waitKey(0)
# cv2.imwrite('affine_img.jpg',affine_img)
cv2.destroyAllWindows()