# 带翻转TTA + 最优阈值 + 简单形态学后处理的推理脚本

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torchvision import transforms as T
import segmentation_models_pytorch as smp

IMAGE_SIZE = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DATA_DIR = './data'
TEST_IMG_DIR = os.path.join(DATA_DIR, 'test_a')
TEST_SAMPLE_CSV = os.path.join(DATA_DIR, 'test_a_samplesubmit.csv')

MODEL_PATH = './best_unetpp_efficientnetb4.pth'
SUBMIT_PATH = './submit_tta.csv'

BEST_THR = 0.5   # 用 search_best_thr.py 输出的最优阈值改掉这里


def rle_encode(im):
    pixels = im.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


trfm = T.Compose([
    T.ToPILImage(),
    T.Resize(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize([0.625, 0.448, 0.688],
                [0.131, 0.177, 0.101]),
])


def predict_prob_with_tta(model, img_rgb):
    """
    img_rgb: HxWx3, RGB np.uint8
    返回: 512x512 的概率图
    """
    # 原图
    imgs = []
    flips = []

    imgs.append(img_rgb)
    flips.append('none')

    imgs.append(img_rgb[:, ::-1, :])   # 水平翻转
    flips.append('h')

    imgs.append(img_rgb[::-1, :, :])   # 垂直翻转
    flips.append('v')

    imgs.append(img_rgb[::-1, ::-1, :])  # 水平+垂直
    flips.append('hv')

    probs_list = []

    with torch.no_grad():
        for im, f in zip(imgs, flips):
            x = trfm(im).to(DEVICE)[None, ...]
            logits = model(x)[0, 0]   # [H,W]
            p = torch.sigmoid(logits).cpu().numpy()

            # 翻转还原
            if f == 'h':
                p = p[:, ::-1]
            elif f == 'v':
                p = p[::-1, :]
            elif f == 'hv':
                p = p[::-1, ::-1]

            probs_list.append(p)

    prob = np.mean(probs_list, axis=0)
    return prob


def main():
    # 1. 加载模型
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    print(f"Loaded model from: {MODEL_PATH}")

    # 2. 读取测试文件名
    test_df = pd.read_csv(TEST_SAMPLE_CSV, sep='\t', names=['name', 'mask'])
    img_names = test_df['name'].values

    kernel = np.ones((3, 3), np.uint8)

    subm = []

    for name in tqdm(img_names, desc="Infer test_a with TTA"):
        img_path = os.path.join(TEST_IMG_DIR, name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        prob = predict_prob_with_tta(model, img)

        # 二值化 + 形态学后处理
        mask = (prob > BEST_THR).astype(np.uint8)

        # 注意：trfm里已经resize成512了，这里只是兜底
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        # 简单开闭运算去小噪点、填小洞
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        rle = rle_encode(mask)
        subm.append([name, rle])

    subm_df = pd.DataFrame(subm)
    subm_df.to_csv(SUBMIT_PATH, index=False, header=False, sep='\t')
    print(f"Saved submit file to: {SUBMIT_PATH}")


if __name__ == '__main__':
    main()
