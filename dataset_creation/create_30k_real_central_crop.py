import os
import random
import cv2
import numpy as np
from tqdm import tqdm

# Paths
INPUT_DIRS = ["30kds/train/Real", "30kds/val/Real", "30kds/test/Real"]
OUTPUT_DIR = "30kds_real_central_crop"
REAL_SUBDIR = os.path.join(OUTPUT_DIR, "Real")

# Create output directory
os.makedirs(REAL_SUBDIR, exist_ok=True)

# Function for central square cropping
def central_square_crop(image):
    height, width = image.shape[:2]
    min_dim = min(width, height)
    
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    
    cropped = image[top:top + min_dim, left:left + min_dim]
    return cropped

# Count total images for progress bar
total_images = 0
for folder in INPUT_DIRS:
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                total_images += 1

print(f"Found {total_images} images to process")

# Process all real images
image_counter = 0
pbar = tqdm(total=total_images, desc="Processing Images")

for folder in INPUT_DIRS:
    if not os.path.exists(folder):
        print(f"Warning: Input directory {folder} does not exist. Skipping.")
        continue
        
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
            # Read image with OpenCV
            img = cv2.imread(src_path)
            
            if img is None:
                pbar.write(f"Error: Could not read {src_path}")
                continue
                
            # Apply central square crop
            img = central_square_crop(img)
            
            # Resize to 128x128
            img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LANCZOS4)
            
            # Save the processed image
            cv2.imwrite(dst_path, img)
            
            image_counter += 1
            pbar.update(1)
                
        except Exception as e:
            pbar.write(f"Error processing {src_path}: {e}")

pbar.close()
print(f"All images processed and saved to {REAL_SUBDIR}")
print(f"Total images processed: {image_counter}")

# For verification, check a few random images
if os.path.exists(REAL_SUBDIR) and len(os.listdir(REAL_SUBDIR)) > 0:
    sample_files = random.sample(os.listdir(REAL_SUBDIR), min(5, len(os.listdir(REAL_SUBDIR))))
    print("\nVerifying dimensions of sample images:")
    for sample in sample_files:
        img_path = os.path.join(REAL_SUBDIR, sample)
        img = cv2.imread(img_path)
        if img is not None:
            height, width = img.shape[:2]
            print(f"{sample}: {width}x{height}")
        else:
            print(f"{sample}: Failed to read")
else:
    print("\nNo images found in output directory for verification.")