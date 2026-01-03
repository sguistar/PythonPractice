# 简化版 Unet++ 训练脚本：用于【地球观察员：建筑物识别学习赛】

import os
import random
import time

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import albumentations as A
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.utils.data as D
from torchvision import transforms as T

import segmentation_models_pytorch as smp

# ================== 基本配置 ==================
EPOCHS = 5
BATCH_SIZE = 4
IMAGE_SIZE = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DATA_DIR = './data'
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train')
TRAIN_MASK_CSV = os.path.join(DATA_DIR, 'train_mask.csv')
MODEL_SAVE_PATH = './best_unetpp_efficientnetb4.pth'


# ================== 随机种子 ==================
def set_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seeds(42)

# ================== RLE 工具函数 ==================
def rle_encode(im):
    """
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = im.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(512, 512)):
    """
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
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


# ================== 数据增强 ==================
train_trfm = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.ColorJitter(brightness=0.07, contrast=0.07,
                      saturation=0.1, hue=0.1, p=0.5),
    ], p=0.5),
])

val_trfm = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
])


# ================== Dataset ==================
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

        img = cv2.imread(img_path)  # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = rle_decode(rle, shape=(512, 512))

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = self.as_tensor(img)  # [3, H, W]
        mask = torch.from_numpy(mask).unsqueeze(0).float()  # [1, H, W]

        return img, mask


# ================== 损失 & 指标 ==================
class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: [B,1,H,W], targets: [B,1,H,W]
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(2, 3)) + self.smooth
        den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + self.smooth
        dice = num / den
        return 1 - dice.mean()


bce_loss = nn.BCEWithLogitsLoss()
dice_loss = SoftDiceLoss()


def criterion(logits, targets, ratio=0.8):
    bce = bce_loss(logits, targets)
    dsc = dice_loss(logits, targets)
    return ratio * bce + (1 - ratio) * dsc


def dice_score_from_logits(logits, targets, thr=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()

    intersection = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2 * intersection + 1e-7) / (union + 1e-7)
    return dice.mean().item()


# ================== 验证 ==================
def validate(model, loader):
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    count = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, masks)
            dice = dice_score_from_logits(logits, masks)

            bs = images.size(0)
            val_loss += loss.item() * bs
            val_dice += dice * bs
            count += bs

    return val_loss / count, val_dice / count


# ================== 主训练逻辑 ==================
def main():
    print("Loading train_mask.csv ...")
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

    # 简单随机划分 train / val
    train_idx, val_idx = train_test_split(
        np.arange(len(img_paths)),
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    train_ds = TianChiDataset(img_paths[train_idx], rles[train_idx], transform=train_trfm)
    val_ds = TianChiDataset(img_paths[val_idx], rles[val_idx], transform=val_trfm)

    train_loader = D.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = D.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    # ================== 模型 ==================
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",   # 会自动加载预训练权重
        in_channels=3,
        classes=1,
    )
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    best_val_loss = float('inf')
    best_val_dice = 0.0

    print("Start training on", DEVICE)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        cnt = 0
        t0 = time.time()

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            dice = dice_score_from_logits(logits, masks)

            bs = images.size(0)
            epoch_loss += loss.item() * bs
            epoch_dice += dice * bs
            cnt += bs

        epoch_loss /= cnt
        epoch_dice /= cnt

        val_loss, val_dice = validate(model, val_loader)
        scheduler.step(val_loss)

        dt = time.time() - t0
        print(f"[Epoch {epoch:03d}] "
              f"train_loss={epoch_loss:.4f} train_dice={epoch_dice:.4f} "
              f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} "
              f"time={dt/60:.1f} min")

        # 根据 val_loss 保存 best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_dice = val_dice
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  >> Saved new best model: val_loss={best_val_loss:.4f}, val_dice={best_val_dice:.4f}")

    print("Training done.")
    print(f"Best val_loss={best_val_loss:.4f}, val_dice={best_val_dice:.4f}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    main()
