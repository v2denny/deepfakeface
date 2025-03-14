import os
import shutil
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image

# Paths
INPUT_DIRS = ["30kds/train/Real", "30kds/val/Real", "30kds/test/Real"]  # All real images
OUTPUT_DIR = "30kds_real"  # Unified directory

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Copy all real images into a single directory
for folder in INPUT_DIRS:
    for filename in os.listdir(folder):
        src_path = os.path.join(folder, filename)
        dst_path = os.path.join(OUTPUT_DIR, filename)
        shutil.copy(src_path, dst_path)

print(f"All real images have been copied to {OUTPUT_DIR}")

# Function for central cropping to make the image square
def central_square_crop(image):
    width, height = image.size
    min_dim = min(width, height)

    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    right = (width + min_dim) / 2
    bottom = (height + min_dim) / 2

    return image.crop((left, top, right, bottom))

# Define transformations for normalization and conversion to tensor
transform = transforms.Compose([
    transforms.Lambda(central_square_crop),  # Apply central crop
    transforms.Resize((256, 256)),  # Resize to 256x256 (change to 512x512 if needed)
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1], common for GANs and VAEs
])

# Create dataset and DataLoader from the consolidated directory
dataset = ImageFolder(root=".", transform=transform)  # '.' since all images are now in OUTPUT_DIR
dataset.samples = [(os.path.join(OUTPUT_DIR, f), 0) for f in os.listdir(OUTPUT_DIR)]  # Adjust paths

# Create DataLoader to feed the model
batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

print("Pipeline successfully loaded! Images are ready for the generative model.")
