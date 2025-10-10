import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread('D:\\PycharmProjects\\class practice\\image1.jpg')
if img1 is None:
    print("Error: Could not load image1.jpg")
    exit()
img2 = cv2.imread('D:\\PycharmProjects\\class practice\\image2.jpg')
if img2 is None:
    print("Error: Could not load image2.jpg")
    exit()

# 将图像转换为灰度图，以便进行单通道处理，如果需要彩色处理，则需要对每个通道分别处理
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# 对数变换 (Image 1) - 调整C值以控制亮度，避免过亮
# 对数变换可以增强图像的暗部细节
img1_float = img1_gray.astype(np.float32)
c1 = 255 / np.log(1 + np.max(img1_float))  # 原始C值，可能导致过亮
img1_log = c1 * np.log(1 + img1_float)
img1_log = np.uint8(np.clip(img1_log, 0, 255))  # 确保值在0-255范围内
img1_eq = cv2.equalizeHist(img1_gray)

# 伽马变换 (Image 2) - 调整gamma值以提亮暗部并控制高光
# 伽马值小于1会提亮暗部，大于1会压暗亮部
img2_float = img2.astype(np.float32) / 255.0
gamma = 0.4  # 减小gamma值以提亮暗部，防止过亮
img2_gamma = np.power(img2_float, gamma)
img2_gamma = np.uint8(np.clip(img2_gamma * 255, 0, 255))
img2_gamma_blur = cv2.GaussianBlur(img2_gamma, (0, 0), sigmaX=3, sigmaY=3)
details = cv2.subtract(img2_gamma, img2_gamma_blur)
img2_sharpened = cv2.addWeighted(img2_gamma, 1.2, details, -0.5, 0)

# 显示原始图像和变换后的图像
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(img1, cmap='gray')
plt.title('Original Image 1 (Gray)')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(img1_eq, cmap='gray')
plt.title('Log Transformed Image 1 (Adjusted C)')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.title('Original Image 2')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(img2_sharpened, cv2.COLOR_BGR2RGB))
plt.title(f'Gamma Transformed Image 2 (Gamma={gamma})')
plt.axis('off')

plt.tight_layout()
# 保存图像到文件
# dpi=300 指定了图像的分辨率（每英寸点数），可以获得更清晰的输出文件
plt.savefig('comparison_results.png', dpi=300, bbox_inches='tight')
plt.show()

# 显示变换后的图像统计信息
print(f"Image 1 - 原始范围: {img1_gray.min()} - {img1_gray.max()}")
print(f"Image 1 - 对数变换后范围: {img1_log.min()} - {img1_log.max()}")
print(f"Image 2 - 原始范围: {img2_gray.min()} - {img2_gray.max()}")
print(f"Image 2 - 伽马变换后范围: {img2_gamma.min()} - {img2_gamma.max()}")
