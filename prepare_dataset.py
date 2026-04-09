from datasets import load_dataset
import matplotlib.pyplot as plt
from dataset_utils import process_sample

data = load_dataset("keremberke/satellite-building-segmentation", name="full")

print("Train size:", len(data["train"]))
print("Validation size:", len(data["validation"]))
print("Test size:", len(data["test"]))

sample_image, sample_mask = process_sample(data["train"][0])

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(sample_image)
plt.title("Image")

plt.subplot(1, 2, 2)
plt.imshow(sample_mask, cmap="gray")
plt.title("Mask (All Houses)")

plt.show(block=True)

print("Done generating one sample mask")