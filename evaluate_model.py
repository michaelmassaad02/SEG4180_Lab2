import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
import matplotlib.pyplot as plt
from dataset_utils import BuildingDataset

data = load_dataset("keremberke/satellite-building-segmentation", name="full")

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(128, 256)

        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv1 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        b = self.bottleneck(p2)

        u1 = self.up1(b)
        u1 = torch.cat([u1, d2], dim=1)
        u1 = self.conv1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.conv2(u2)

        return self.final(u2)

def compute_iou(pred_mask, true_mask, smooth=1e-6):
    intersection = (pred_mask * true_mask).sum()
    union = pred_mask.sum() + true_mask.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def compute_dice(pred_mask, true_mask, smooth=1e-6):
    intersection = (pred_mask * true_mask).sum()
    return (2 * intersection + smooth) / (pred_mask.sum() + true_mask.sum() + smooth)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = UNet().to(device)
model.load_state_dict(torch.load("house_segmentation_model.pth", map_location=device))
model.eval()

test_dataset = BuildingDataset(data["test"], max_samples=20)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

ious = []
dices = []

sample_images = []
sample_true_masks = []
sample_pred_masks = []

with torch.no_grad():
    for i, (images, masks) in enumerate(test_loader):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()

        pred_mask = preds.squeeze().cpu().numpy()
        true_mask = masks.squeeze().cpu().numpy()
        image_np = images.squeeze().cpu().permute(1, 2, 0).numpy()

        iou = compute_iou(pred_mask, true_mask)
        dice = compute_dice(pred_mask, true_mask)

        ious.append(iou)
        dices.append(dice)

        if i < 3:
            sample_images.append(image_np)
            sample_true_masks.append(true_mask)
            sample_pred_masks.append(pred_mask)

avg_iou = np.mean(ious)
avg_dice = np.mean(dices)

print(f"Average IoU: {avg_iou:.4f}")
print(f"Average Dice Score: {avg_dice:.4f}")

for i in range(len(sample_images)):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(sample_images[i])
    plt.title("Aerial Image")

    plt.subplot(1, 3, 2)
    plt.imshow(sample_true_masks[i], cmap="gray")
    plt.title("Ground Truth Mask")

    plt.subplot(1, 3, 3)
    plt.imshow(sample_pred_masks[i], cmap="gray")
    plt.title("Predicted Mask")

    plt.tight_layout()
    plt.show()