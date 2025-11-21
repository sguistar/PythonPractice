import cv2
import numpy as np

img = cv2.imread('test.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 200, 300)
cv2.imshow("edge",edges)
cv2.waitKey(0)
circles = cv2.HoughCircles(
    image=edges,
    method=cv2.HOUGH_GRADIENT,
    dp=1.2,               # 累加器分辨率1.2（比图像分辨率低，速度更快）
    minDist=50,           # 两圆心最小距离50像素（避免重叠检测）
    param1=100,           # Canny高阈值100（低阈值=50）
    param2=41,            # 累加器阈值50（仅保留强投票圆）
    minRadius=20,         # 最小半径20像素（过滤小噪声）
    maxRadius=80          # 最大半径80像素（过滤大干扰）
)

if circles is not None:
    # 转换为整数坐标（避免绘图报错）
    circles = np.uint16(np.around(circles))
    for circle in circles[0]:
        x, y, r = circle[0], circle[1], circle[2]
        # 绘制圆心（红色，半径2像素）
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
        # 绘制圆轮廓（绿色，线宽2像素）
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)

cv2.imshow("lines", img)
cv2.waitKey(0)
cv2.imwrite('best_result.jpg', img)
