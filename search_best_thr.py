# 在验证集上搜索最优阈值

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as D
from torchvision import transforms as T
import albumentations as A
import segmentation_models_pytorch as smp

# ========= 和 train_simple.py 保持一致 =========
IMAGE_SIZE = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DATA_DIR = './data'
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train')
TRAIN_MASK_CSV = os.path.join(DATA_DIR, 'train_mask.csv')
MODEL_PATH = './best_unetpp_efficientnetb4.pth'


def rle_decode(mask_rle, shape=(512, 512)):
    if (mask_rle is np.nan) or (mask_rle is None) or (mask_rle == ''):
        return np.zeros(shape[0] * shape[1], dtype=np.uint8).reshape(shape, order='F')
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


val_trfm = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
])


class TianChiDataset(D.Dataset):
    def __init__(self, img_paths, rles, transform=None):
        self.img_paths = img_paths
        self.rles = rles
        self.transform = transform

        self.as_tensor = T.Compose([
            T.ToPILImage(),
            T.Resize(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize([0.625, 0.448, 0.688],
                        [0.131, 0.177, 0.101]),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        rle = self.rles[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = rle_decode(rle, (512, 512))

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = self.as_tensor(img)
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return img, mask


def main():
    # 1. 重新读取数据并划分 val（保证 random_state 一致）
    train_df = pd.read_csv(TRAIN_MASK_CSV, sep='\t', names=['name', 'mask'])
    train_df['img_path'] = train_df['name'].apply(lambda x: os.path.join(TRAIN_IMG_DIR, x))

    # 只保留“图像文件确实存在”的行
    train_df['exists'] = train_df['img_path'].apply(os.path.exists)
    missing = (~train_df['exists']).sum()
    if missing > 0:
        print(f"[WARN] {missing} images listed in CSV but file not found, they will be dropped.")
    train_df = train_df[train_df['exists']].drop(columns=['exists']).reset_index(drop=True)

    img_paths = train_df['img_path'].values
    rles = train_df['mask'].fillna('').values

    train_idx, val_idx = train_test_split(
        np.arange(len(img_paths)),
        test_size=0.2,
        random_state=42,  # 和训练脚本里保持一致
        shuffle=True
    )

    val_ds = TianChiDataset(img_paths[val_idx], rles[val_idx], transform=val_trfm)
    val_loader = D.DataLoader(val_ds, batch_size=4, shuffle=False,
                              num_workers=0, pin_memory=False)

    # 2. 加载模型
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

    # 3. 对一组阈值做全量遍历
    thrs = np.linspace(0.3, 0.7, 21)  # 0.30, 0.32, ..., 0.70
    inter_sums = np.zeros_like(thrs)
    union_sums = np.zeros_like(thrs)

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Eval val for thr search"):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)[:, 0, :, :]          # [B,H,W]
            probs = torch.sigmoid(logits).cpu().numpy()
            gts = masks[:, 0, :, :].cpu().numpy()

            B = probs.shape[0]
            for i in range(B):
                p = probs[i]
                gt = gts[i]
                gt = (gt > 0.5).astype(np.uint8)

                for k, thr in enumerate(thrs):
                    pred = (p > thr).astype(np.uint8)
                    inter = (pred * gt).sum()
                    union = pred.sum() + gt.sum()
                    inter_sums[k] += inter
                    union_sums[k] += union

    dices = (2 * inter_sums + 1e-7) / (union_sums + 1e-7)

    for t, d in zip(thrs, dices):
        print(f"thr={t:.2f}, dice={d:.4f}")

    best_idx = np.argmax(dices)
    best_thr = float(thrs[best_idx])
    best_dice = float(dices[best_idx])
    print("=" * 50)
    print(f"Best thr = {best_thr:.3f}, val dice = {best_dice:.4f}")


if __name__ == "__main__":
    main()
