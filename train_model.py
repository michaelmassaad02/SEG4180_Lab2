import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
import matplotlib.pyplot as plt
from dataset_utils import BuildingDataset
from model import UNet

# Load dataset from Hugging Face
data = load_dataset("keremberke/satellite-building-segmentation", name="full")

# Select device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Create training and validation datasets
# max_samples is used to limit dataset size for faster training
train_dataset = BuildingDataset(data["train"], max_samples=1200, augment=True)
val_dataset = BuildingDataset(data["validation"], max_samples=300, augment=False)

# Create DataLoaders for batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Initialize UNet model with dropout
model = UNet(dropout=0.4).to(device)

# Define optimizer (Adam) with learning rate
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Learning rate scheduler to reduce LR when validation loss plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=3, factor=0.5
)

# Binary Cross Entropy loss (used for pixel-wise classification)
bce_loss = nn.BCEWithLogitsLoss()

def dice_loss(pred, target, smooth=1e-6):
    """
    Computes Dice loss for segmentation.
    Encourages overlap between predicted and true masks.
    """
    pred = torch.sigmoid(pred) # Convert logits to probabilities
    intersection = (pred * target).sum(dim=(2, 3))
    denom = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return (1 - (2 * intersection + smooth) / (denom + smooth)).mean()

def combined_loss(pred, target):
    """
    Combines BCE loss and Dice loss to improve segmentation performance.
    """
    return bce_loss(pred, target) + dice_loss(pred, target)

# Lists to track loss values over epochs
train_losses = []
val_losses = []
best_val_loss = float("inf")

# Number of training epochs
num_epochs = 8

for epoch in range(num_epochs):
    model.train() # Set model to training mode
    running_train_loss = 0.0

    # Iterate over training batches
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad() # Reset gradients
        outputs = model(images) # Forward pass
        loss = combined_loss(outputs, masks) # Compute loss
        loss.backward() # Backpropagation
        optimizer.step()  # Update model weights

        running_train_loss += loss.item()

    # Compute average training loss
    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval() # Set model to evaluation mode
    running_val_loss = 0.0

    with torch.no_grad(): # Disable gradients for validation
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = combined_loss(outputs, masks)
            running_val_loss += loss.item()
    
    # Compute average validation loss
    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # Update learning rate based on validation loss
    scheduler.step(avg_val_loss)

    # Print training progress
    print(
        f"Epoch {epoch+1:>2}/{num_epochs}, "
        f"Train Loss: {avg_train_loss:.4f}, "
        f"Val Loss: {avg_val_loss:.4f}"
    )

    # Save model if validation loss improves
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "house_segmentation_model.pth")
        print("Best model updated and saved.")

print("Training finished.")

# Visualize training and validation loss over epochs
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=300, bbox_inches="tight")
plt.show()