from datasets import load_dataset
import matplotlib.pyplot as plt
from dataset_utils import process_sample

# Load the satellite building segmentation dataset from Hugging Face
data = load_dataset("keremberke/satellite-building-segmentation", name="full")

# Print dataset split sizes
print("Train size:", len(data["train"]))
print("Validation size:", len(data["validation"]))
print("Test size:", len(data["test"]))

# Process a single sample from the training set
# This converts bounding boxes into a combined segmentation mask
sample_image, sample_mask = process_sample(data["train"][0])

# Create a figure to display the image and its corresponding mask
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(sample_image)
plt.title("Image")

plt.subplot(1, 2, 2)
plt.imshow(sample_mask, cmap="gray")
plt.title("Mask (All Houses)")

plt.show(block=True)

print("Done generating one sample mask")