import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import matplotlib.pyplot as plt
from dataset_utils import BuildingDataset
from model import UNet

# Load the dataset from Hugging Face
data = load_dataset("keremberke/satellite-building-segmentation", name="full")

def compute_iou(pred_mask, true_mask, smooth=1e-6):
    """
    Computes Intersection over Union (IoU) between predicted and true masks.
    IoU = intersection / union
    """
    intersection = (pred_mask * true_mask).sum()
    union = pred_mask.sum() + true_mask.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def compute_dice(pred_mask, true_mask, smooth=1e-6):
    """
    Computes Dice score between predicted and true masks.
    Dice = 2 * intersection / (sum of masks)
    """
    intersection = (pred_mask * true_mask).sum()
    return (2 * intersection + smooth) / (pred_mask.sum() + true_mask.sum() + smooth)

# Select device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load trained UNet model
model = UNet().to(device)
model.load_state_dict(torch.load("house_segmentation_model.pth", map_location=device))
model.eval() # Set model to evaluation mode

# Prepare test dataset and dataloader
test_dataset = BuildingDataset(data["test"], augment=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Lists to store evaluation metrics
ious = []
dices = []

# Lists to store sample predictions for visualization
sample_images = []
sample_true_masks = []
sample_pred_masks = []

# Disable gradient computation for evaluation (faster and memory-efficient)
with torch.no_grad():
    for i, (images, masks) in enumerate(test_loader):
        images = images.to(device)
        masks = masks.to(device)

        # Run model inference
        outputs = model(images)
        probs = torch.sigmoid(outputs) # Convert logits to probabilities
        preds = (probs > 0.5).float() # Apply threshold to get binary mask

        # Convert tensors to numpy for metric calculation
        pred_mask = preds.squeeze().cpu().numpy()
        true_mask = masks.squeeze().cpu().numpy()
        image_np = images.squeeze().cpu().permute(1, 2, 0).numpy()
        
        # Compute evaluation metrics
        ious.append(compute_iou(pred_mask, true_mask))
        dices.append(compute_dice(pred_mask, true_mask))

        # Store first 5 samples for visualization
        if i < 5:
            sample_images.append(image_np)
            sample_true_masks.append(true_mask)
            sample_pred_masks.append(pred_mask)

# Compute average IoU and Dice score across the test set
avg_iou = np.mean(ious)
avg_dice = np.mean(dices)

print(f"Average IoU: {avg_iou:.4f}")
print(f"Average Dice Score: {avg_dice:.4f}")

# Visualize and save prediction results
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