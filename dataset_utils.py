import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def make_mask(labelled_bbox, image):
    """
    Converts a single bounding box into a binary mask.
    Pixels inside the bounding box are set to 1 (building),
    and all other pixels remain 0 (background).
    """
    x_min, y_min, width, height = labelled_bbox
    x_min, y_min, width, height = int(x_min), int(y_min), int(width), int(height)

    # Initialize empty mask (all background)
    mask = np.zeros((image.height, image.width), dtype=np.uint8)

    # Compute bottom-right corner of bounding box
    x_max = x_min + width
    y_max = y_min + height

    # Fill bounding box area with 1s (building region)
    mask[y_min:y_max, x_min:x_max] = 1
    return mask

def process_sample(sample):
    """
    Processes a dataset sample by:
    - extracting the image
    - converting all bounding boxes into masks
    - combining them into a single segmentation mask
    """
    image = sample["image"]
    bboxes = sample["objects"]["bbox"]

    # Initialize empty mask for the full image
    combined_mask = np.zeros((image.height, image.width), dtype=np.uint8)

    # Generate mask for each bounding box and merge them
    for bbox in bboxes:
        mask = make_mask(bbox, image)
        combined_mask = np.maximum(combined_mask, mask)

    return image, combined_mask

def apply_augmentation(image, mask):
    """
    Applies random data augmentation to both the image and mask.
    Ensures transformations are applied consistently to both.
    """
     
    # Horizontal flip   
    if random.random() > 0.5:
        image = transforms.functional.hflip(image)
        mask = np.fliplr(mask)

    # Vertical flip
    if random.random() > 0.5:
        image = transforms.functional.vflip(image)
        mask = np.flipud(mask)

    # Random rotation (90, 180, 270 degrees)
    if random.random() > 0.5:
        angle = random.choice([90, 180, 270])
        image = transforms.functional.rotate(image, angle)
        mask = np.rot90(mask, k=angle // 90)

    # Random color jitter (applied only to image)
    if random.random() > 0.5:
        image = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)(image)
    return image, mask


class BuildingDataset(Dataset):
    """
    Custom PyTorch Dataset for building segmentation.
    Handles:
    - mask generation from bounding boxes
    - resizing
    - optional data augmentation
    """
    def __init__(self, split, max_samples=None, image_size=(256, 256), augment = False):
        """
        Args:
            split: dataset split (train/validation/test)
            max_samples: limit number of samples (for faster training)
            image_size: target size for images and masks
            augment: whether to apply data augmentation
        """
        self.split = split
        self.max_samples = max_samples if max_samples is not None else len(split)
        self.image_size = image_size
        self.augment = augment

        # Define image transformation pipeline
        self.image_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return min(self.max_samples, len(self.split))

    def __getitem__(self, idx):
        """
        Retrieves one sample (image + mask) from the dataset.
        """
        sample = self.split[idx]

        # Generate image and combined mask from bounding boxes
        image, mask = process_sample(sample)

        # Resize image and mask
        image = image.resize(self.image_size, Image.BILINEAR)
        mask = Image.fromarray(mask).resize(self.image_size, Image.NEAREST)
        mask = np.array(mask)
        
        # Apply augmentation if enabled
        if self.augment:
            image, mask = apply_augmentation(image, mask)

        # Convert image to tensor
        image = self.image_transform(image)

        # Convert mask to tensor (add channel dimension)
        mask = torch.tensor(mask.copy(), dtype=torch.float32).unsqueeze(0)

        return image, mask