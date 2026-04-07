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

class BuildingDataset(Dataset):
    def __init__(self, split, max_samples=None, image_size=(256, 256)):
        self.split = split
        self.max_samples = max_samples if max_samples is not None else len(split)
        self.image_size = image_size

        self.image_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return min(self.max_samples, len(self.split))

    def __getitem__(self, idx):
        sample = self.split[idx]
        image, mask = process_sample(sample)

        image = self.image_transform(image)

        mask = Image.fromarray(mask)
        mask = mask.resize(self.image_size, Image.NEAREST)
        mask = torch.tensor(np.array(mask), dtype=torch.float32).unsqueeze(0)

        return image, mask