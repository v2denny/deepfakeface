import os
import shutil
import random
from PIL import Image
from tqdm import tqdm  # For progress bars (optional)
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Paths
INPUT_DIRS = ["30kds/train/Real", "30kds/val/Real", "30kds/test/Real"]
OUTPUT_DIR = "30kds_real"
REAL_SUBDIR = os.path.join(OUTPUT_DIR, "Real")

# Create output directory
os.makedirs(REAL_SUBDIR, exist_ok=True)

# Function for central square cropping
def central_square_crop(image):
    width, height = image.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    return image.crop((left, top, right, bottom))

# Process all real images
total_images = 0
for folder in INPUT_DIRS:
    total_images += len(os.listdir(folder))

print(f"Processing {total_images} images...")

image_counter = 0
for folder in INPUT_DIRS:
    for filename in os.listdir(folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        src_path = os.path.join(folder, filename)
        
        # Generate unique filename
        base, ext = os.path.splitext(filename)
        dst_filename = filename
        counter = 1
        while os.path.exists(os.path.join(REAL_SUBDIR, dst_filename)):
            dst_filename = f"{base}_{counter}{ext}"
            counter += 1
        
        dst_path = os.path.join(REAL_SUBDIR, dst_filename)
        
        try:
            # Open, process, and save the image
            with Image.open(src_path) as img:
                # Apply central square crop
                img = central_square_crop(img)
                # Resize to 256x256
                img = img.resize((256, 256), Image.LANCZOS)
                # Save the processed image
                img.save(dst_path)
            
            image_counter += 1
            if image_counter % 100 == 0:
                print(f"Processed {image_counter}/{total_images} images")
                
        except Exception as e:
            print(f"Error processing {src_path}: {e}")

print(f"All images processed and saved to {REAL_SUBDIR}")
print(f"Total images processed: {image_counter}")

# For verification, check a few random images
sample_files = random.sample(os.listdir(REAL_SUBDIR), min(5, len(os.listdir(REAL_SUBDIR))))
print("\nVerifying dimensions of sample images:")
for sample in sample_files:
    img_path = os.path.join(REAL_SUBDIR, sample)
    with Image.open(img_path) as img:
        print(f"{sample}: {img.size}")

# Set up dataset with normalization for training
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize RGB to [-1, 1]
])

# Create dataset and DataLoader
dataset = ImageFolder(root=OUTPUT_DIR, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

print("\nDataset setup complete!")
print(f"Number of images in dataset: {len(dataset)}")
print("Images are ready for the generative model.")