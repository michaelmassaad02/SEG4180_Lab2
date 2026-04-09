import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import matplotlib.pyplot as plt
from dataset_utils import BuildingDataset
from model import UNet

data = load_dataset("keremberke/satellite-building-segmentation", name="full")

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

test_dataset = BuildingDataset(data["test"], augment=False)
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

        ious.append(compute_iou(pred_mask, true_mask))
        dices.append(compute_dice(pred_mask, true_mask))

        if i < 5:
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
    plt.savefig(f"prediction_example_{i+1}.png", dpi=300, bbox_inches='tight')
    plt.show()