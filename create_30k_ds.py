import os
import cv2
import random
import shutil
from tqdm import tqdm

# Paths to your datasets
DATASETS = {
    "wiki": "wiki",  # Real images
    "inpainting": "inpainting",  # Fake images
    "insight": "insight",  # Fake images
    "text2img": "text2img",  # Fake images
}

# YOLO output structure
YOLO_DATASET_PATH = "30kds/"
SPLITS = ["train", "test", "val"]
CLASS_NAMES = ["Real", "Fake"]
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1  # 80% Train, 10% Val, 10% Test

def count_images_in_dataset(dataset_path):
    """Counts the number of images in the given dataset folder."""
    image_extensions = ('.jpg', '.png', '.jpeg')
    total_images = 0

    for folder in range(100):  # Your dataset has subfolders from 0 to 99
        folder_path = os.path.join(dataset_path, str(folder))
        if os.path.exists(folder_path):
            images = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
            total_images += len(images)

    return total_images

def create_yolo_structure():
    """Create YOLO dataset folder structure."""
    for split in SPLITS:
        for class_name in CLASS_NAMES:
            os.makedirs(os.path.join(YOLO_DATASET_PATH, split, class_name), exist_ok=True)

def get_images(dataset_path, sample_size):
    """Get a random sample of images from the dataset."""
    all_images = []
    for folder in range(100):  # Your datasets have folders 0 to 99
        folder_path = os.path.join(dataset_path, str(folder))
        if os.path.exists(folder_path):
            images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
            all_images.extend(images)

    return random.sample(all_images, min(sample_size, len(all_images)))

def process_and_save_images(images, class_name):
    """Process and save images one by one to avoid memory issues."""
    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    dataset_splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, img_list in dataset_splits.items():
        for img_path in tqdm(img_list, desc=f"Processing {class_name} for {split}"):
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Save image in YOLO dataset structure
            filename = f"{random.randint(10000, 99999)}_{class_name}.png"
            dest_path = os.path.join(YOLO_DATASET_PATH, split, class_name, filename)
            shutil.copy(img_path, dest_path)

def main():
    """Main function to create the YOLO dataset."""
    print("Creating YOLO dataset structure...")
    create_yolo_structure()

    print("Counting real images in 'wiki' dataset...")
    real_image_count = count_images_in_dataset(DATASETS["wiki"])
    print(f"Total real images found: {real_image_count}")

    print("Processing real images...")
    real_images = get_images(DATASETS["wiki"], real_image_count)  # Get all real images
    process_and_save_images(real_images, "Real")

    print(f"Processing fake images to match {real_image_count} real images...")
    fake_images = []
    num_fake_per_dataset = real_image_count // 3  # Distribute equally among 3 fake datasets

    for fake_ds in ["inpainting", "insight", "text2img"]:
        fake_images.extend(get_images(DATASETS[fake_ds], num_fake_per_dataset))

    # If there's a remainder, add a few extra images from the first fake dataset
    remaining_fake_images = real_image_count - len(fake_images)
    if remaining_fake_images > 0:
        fake_images.extend(get_images(DATASETS["inpainting"], remaining_fake_images))

    process_and_save_images(fake_images, "Fake")

    print("Dataset processing complete!")

if __name__ == "__main__":
    main()
