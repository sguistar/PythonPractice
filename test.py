import torch
import segmentation_models_pytorch as smp

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 建一个小一点的 Unet++，图像也缩小
model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b4",
    encoder_weights=None,   # 先不用加载imagenet，避免第一次下载/初始化太慢
    in_channels=3,
    classes=1,
).to(device)

x = torch.randn(1, 3, 256, 256).to(device)

with torch.no_grad():
    y = model(x)

print("output shape:", y.shape)
