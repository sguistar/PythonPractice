import cv2
import numpy as np
from PIL import Image

# 1. 加载人脸检测器
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 2. 加载关键点检测器
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel(r"D:\PycharmProjects\class practice\lbfmodel.yaml")

# ------------------ 参考人脸 ------------------
image1 = cv2.imread(r"D:\PycharmProjects\class practice\face1.jpg")
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

faces1 = face_detector.detectMultiScale(gray1, 1.3, 5)
_, landmarks1 = facemark.fit(gray1, faces1)
lm1 = landmarks1[0][0]   # 第一个人脸的关键点

# 提取关键点（眼睛和鼻子）
refpoints = np.float32([
    lm1[36],  # 左眼角
    lm1[45],  # 右眼角
    lm1[30]   # 鼻尖
])
print("refpoints:", refpoints)

h, w = image1.shape[:2]

# ------------------ 待对齐人脸 ------------------
image2 = cv2.imread(r"new.jpg")
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

faces2 = face_detector.detectMultiScale(gray2, 1.25, 5)

print(f"Detected {len(faces2)} faces")
_, landmarks2 = facemark.fit(gray2, faces2)

aligned_faces = []
for i, rect in enumerate(faces2):
    if len(landmarks2) <= i or landmarks2[i] is None:
        print(f"Skipping face {i+1}: no landmarks found.")
        continue

    lm2 = landmarks2[i][0]
    points = np.float32([
        lm2[36],
        lm2[45],
        lm2[30]
    ])
    print("points:", points)

    # ------------------ 刚性变换 ------------------
    M, _ = cv2.estimateAffinePartial2D(points, refpoints)
    aligned = cv2.warpAffine(image2, M, (w, h))
    # 安全裁剪（用原图坐标）
    x, y, fw, fh = rect
    x, y = max(0, x), max(0, y)
    fw, fh = min(fw, image2.shape[1] - x), min(fh, image2.shape[0] - y)
    cropped = image2[y:y+fh, x:x+fw]
    if cropped.size == 0:
        print(f"Skipping face {i+1}: cropped area empty.")
        continue
    aligned_faces.append(cropped)
    cv2.imwrite(f'aligned_face_{i+1}.jpg', cropped)

if aligned_faces:
    # 将所有裁剪结果缩放为统一大小
    resized_faces = [cv2.resize(face, (200, 200)) for face in aligned_faces]
    combined = cv2.hconcat(resized_faces)
    cv2.imwrite('new_adjusted.jpg', combined)
    print("All faces saved to new_adjusted.jpg")
else:
    print("No faces detected.")

# ------------------ 显示结果 ------------------
# aligned_pil = Image.fromarray(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
# aligned_pil.show()
