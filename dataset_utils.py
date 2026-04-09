import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def make_mask(labelled_bbox, image):
    x_min, y_min, width, height = labelled_bbox
    x_min, y_min, width, height = int(x_min), int(y_min), int(width), int(height)

    mask = np.zeros((image.height, image.width), dtype=np.uint8)

    x_max = x_min + width
    y_max = y_min + height

    mask[y_min:y_max, x_min:x_max] = 1
    return mask

def process_sample(sample):
    image = sample["image"]
    bboxes = sample["objects"]["bbox"]

    combined_mask = np.zeros((image.height, image.width), dtype=np.uint8)

    for bbox in bboxes:
        mask = make_mask(bbox, image)
        combined_mask = np.maximum(combined_mask, mask)

    return image, combined_mask

def apply_augmentation(image, mask):
    if random.random() > 0.5:
        image = transforms.functional.hflip(image)
        mask = np.fliplr(mask)

    if random.random() > 0.5:
        image = transforms.functional.vflip(image)
        mask = np.flipud(mask)

    if random.random() > 0.5:
        angle = random.choice([90, 180, 270])
        image = transforms.functional.rotate(image, angle)
        mask = np.rot90(mask, k=angle // 90)

    if random.random() > 0.5:
        image = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)(image)
    return image, mask


class BuildingDataset(Dataset):
    def __init__(self, split, max_samples=None, image_size=(256, 256), augment = False):
        self.split = split
        self.max_samples = max_samples if max_samples is not None else len(split)
        self.image_size = image_size
        self.augment = augment

        self.image_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return min(self.max_samples, len(self.split))

    def __getitem__(self, idx):
        sample = self.split[idx]
        image, mask = process_sample(sample)

        image = image.resize(self.image_size, Image.BILINEAR)
        mask = Image.fromarray(mask).resize(self.image_size, Image.NEAREST)
        mask = np.array(mask)

        if self.augment:
            image, mask = apply_augmentation(image, mask)

        image = self.image_transform(image)
        mask = torch.tensor(mask.copy(), dtype=torch.float32).unsqueeze(0)

        return image, mask