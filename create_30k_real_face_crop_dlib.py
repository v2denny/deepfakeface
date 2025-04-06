import os
import random
import cv2
import numpy as np
import dlib
from tqdm import tqdm

# Global variables
LIMIT = 0  # Limit processing to specified number of images (0 = all images)
DEBUG = False  # Enable debug mode with verbose output
VISUALIZE = False  # Save debug visualization of detected faces
IMAGE_LIMIT = LIMIT  # Using LIMIT for consistency

# Paths
INPUT_DIRS = ["30kds/train/Real", "30kds/val/Real", "30kds/test/Real"]
OUTPUT_DIR = "30kds_real_face_crop_dlib"
REAL_SUBDIR = os.path.join(OUTPUT_DIR, "Real")
DEBUG_DIR = os.path.join(OUTPUT_DIR, "Debug") if VISUALIZE else None

# Create output directories
os.makedirs(REAL_SUBDIR, exist_ok=True)
if VISUALIZE:
    os.makedirs(DEBUG_DIR, exist_ok=True)

# Initialize face detectors
print("Loading face detectors...")

# Initialize Dlib's face detector and shape predictor
dlib_detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"  # Make sure to download this file
try:
    landmark_predictor = dlib.shape_predictor(predictor_path)
    landmarks_available = True
    print("Dlib face detector and landmark predictor loaded successfully")
except Exception as e:
    landmarks_available = False
    print(f"Dlib face detector loaded, but landmark predictor not available: {e}")
    print("Download the shape predictor from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")

# Function to convert dlib rectangle to OpenCV format
def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

# Improved function to get better bounding box using facial landmarks
def get_improved_bbox(shape, orig_bbox, img_shape):
    # Convert shape to numpy array
    shape_np = np.zeros((68, 2), dtype=int)
    for i in range(68):
        shape_np[i] = (shape.part(i).x, shape.part(i).y)
    
    # Get face boundaries from landmarks
    min_x = np.min(shape_np[:, 0])
    max_x = np.max(shape_np[:, 0])
    min_y = np.min(shape_np[:, 1])
    max_y = np.max(shape_np[:, 1])
    
    # Calculate center of face
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    
    # Calculate width and height of face based on landmarks
    # Add a bit more margin for forehead since landmarks don't capture hairline
    width = int((max_x - min_x) * 1.1)  # 10% wider
    height = int((max_y - min_y) * 1.2)  # 20% taller to account for forehead
    
    # Ensure minimum size (use original bbox if it's larger)
    x, y, w, h = orig_bbox
    width = max(width, w)
    height = max(height, h)
    
    # Create a square bounding box centered on the face
    square_size = max(width, height)
    
    # Return as x, y, width, height
    return (center_x - square_size//2, center_y - square_size//2, square_size, square_size)

# Function for enhanced face detection using Dlib with multiple detection passes
def detect_face(image, filename=""):
    height, width = image.shape[:2]
    faces = []
    
    # Convert to grayscale for Dlib
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # First detection pass - normal image
    dlib_faces = dlib_detector(gray, 1)  # 1 means upsample once for better detection
    
    if dlib_faces:
        for face in dlib_faces:
            bbox = rect_to_bb(face)
            confidence = 0.90  # Base confidence
            faces.append((*bbox, confidence, "Dlib-Normal", face))
    
    # If no faces found, try with histogram equalization
    if not faces:
        equalized = cv2.equalizeHist(gray)
        dlib_faces = dlib_detector(equalized, 1)
        for face in dlib_faces:
            bbox = rect_to_bb(face)
            confidence = 0.85  # Slightly lower confidence for equalized image
            faces.append((*bbox, confidence, "Dlib-Equalized", face))
    
    # If still no faces, try with different scales and rotations
    if not faces:
        # Try different rotations
        for angle in [-15, -5, 5, 15]:
            M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
            rotated = cv2.warpAffine(gray, M, (width, height))
            dlib_faces = dlib_detector(rotated, 1)
            
            if dlib_faces:
                # Need to transform the coordinates back
                for face in dlib_faces:
                    # Convert dlib rectangle to points
                    x, y, w, h = rect_to_bb(face)
                    
                    # Transform the points back to original image coordinates
                    # This is a simplified approach - for precise transforms, use proper inverse rotation
                    # Get center point
                    center_x = x + w//2
                    center_y = y + h//2
                    
                    # Rotate back
                    M_inv = cv2.getRotationMatrix2D((width/2, height/2), -angle, 1)
                    orig_center_x = int(M_inv[0, 0] * center_x + M_inv[0, 1] * center_y + M_inv[0, 2])
                    orig_center_y = int(M_inv[1, 0] * center_x + M_inv[1, 1] * center_y + M_inv[1, 2])
                    
                    # Adjust box size
                    orig_x = orig_center_x - w//2
                    orig_y = orig_center_y - h//2
                    
                    # Add to faces with lower confidence (since it's from a rotated image)
                    confidence = 0.80
                    # Create a new Dlib rect object for the face
                    dlib_rect = dlib.rectangle(
                        left=max(0, orig_x),
                        top=max(0, orig_y),
                        right=min(width, orig_x + w),
                        bottom=min(height, orig_y + h)
                    )
                    faces.append((orig_x, orig_y, w, h, confidence, "Dlib-Rotated", dlib_rect))
    
    if DEBUG:
        print(f"\nDetection results for {filename}:")
        print(f"  Found {len(faces)} faces")
        for i, (x, y, w, h, conf, method, _) in enumerate(faces):
            print(f"  Face {i+1}: method={method}, confidence={conf:.3f}, coords=[{x}, {y}, {w}, {h}]")
    
    # Find the best face (highest confidence, or largest area if same confidence)
    if not faces:
        if DEBUG:
            print("  No face detected")
        return None
    
    # First sort by confidence, then by area if confidence is the same
    best_face_data = max(faces, key=lambda f: (f[4], f[2] * f[3]))
    x, y, w, h, conf, method, dlib_rect = best_face_data
    
    # Improve bounding box using landmarks if available
    if landmarks_available:
        try:
            shape = landmark_predictor(gray, dlib_rect)
            improved_bbox = get_improved_bbox(shape, (x, y, w, h), (height, width))
            
            if DEBUG:
                print(f"  Improved bbox with landmarks: {improved_bbox}")
            
            return improved_bbox
        except Exception as e:
            if DEBUG:
                print(f"  Error using landmarks: {e}")
            
    # Return basic bounding box if landmarks failed or unavailable
    return (x, y, w, h)

# Function for central square cropping (fallback)
def central_square_crop(image):
    height, width = image.shape[:2]
    min_dim = min(width, height)
    
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    
    cropped = image[top:top + min_dim, left:left + min_dim]
    return cropped

# Improved subject-based cropping with better margins
def subject_based_crop(image, box):
    height, width = image.shape[:2]
    
    # Extract box coordinates
    x, y, box_width, box_height = box
    
    # Add margin around the subject (50% of box size)
    margin_x = int(box_width * 0.5)
    margin_y = int(box_height * 0.5)
    
    # Calculate crop boundaries with margins
    left = max(0, x - margin_x)
    top = max(0, y - margin_y)
    right = min(width, x + box_width + margin_x)
    bottom = min(height, y + box_height + margin_y)
    
    # Make it square by taking the larger dimension
    crop_size = max(right - left, bottom - top)
    
    # Calculate new center point
    center_x = left + (right - left) // 2
    center_y = top + (bottom - top) // 2
    
    # Adjust crop boundaries to maintain square and keep centered on subject
    half_size = crop_size // 2
    left = max(0, center_x - half_size)
    top = max(0, center_y - half_size)
    
    # Adjust if we go out of bounds
    if left + crop_size > width:
        left = max(0, width - crop_size)
    if top + crop_size > height:
        top = max(0, height - crop_size)
    
    # Crop the image
    cropped = image[top:min(top + crop_size, height), left:min(left + crop_size, width)]
    
    if DEBUG:
        print(f"  Cropped region: [{left}, {top}, {crop_size}, {crop_size}]")
    
    return cropped

# Function to save debug visualization
def save_debug_visualization(image, box, filename):
    if not VISUALIZE:
        return
        
    # Create a copy for visualization
    vis_img = image.copy()
    
    if box is not None:
        # Draw the detection box
        x, y, w, h = box
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add label with coordinates
        label = f"Face: ({x},{y},{w},{h})"
        cv2.putText(vis_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        # Add "No detection" label
        cv2.putText(vis_img, "No face detected - using central crop", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Save the visualization
    debug_path = os.path.join(DEBUG_DIR, f"debug_{filename}")
    cv2.imwrite(debug_path, vis_img)

# Count total images for progress bar
total_images = 0
for folder in INPUT_DIRS:
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                total_images += 1
                if IMAGE_LIMIT > 0 and total_images >= IMAGE_LIMIT:
                    break

# Adjust total if limit is set
if IMAGE_LIMIT > 0:
    total_images = min(total_images, IMAGE_LIMIT)

print(f"Found {total_images} images to process")

# Process images
image_counter = 0
face_cropped = 0
central_cropped = 0
pbar = tqdm(total=total_images, desc="Processing Images")

for folder in INPUT_DIRS:
    if not os.path.exists(folder):
        print(f"Warning: Input directory {folder} does not exist. Skipping.")
        continue
        
    for filename in os.listdir(folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        # Check if we've reached the limit
        if IMAGE_LIMIT > 0 and image_counter >= IMAGE_LIMIT:
            break
            
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
            
            if DEBUG:
                print(f"\nProcessing {src_path}")
                print(f"  Image size: {img.shape[1]}x{img.shape[0]}")
            
            # Try to detect face using our improved method
            face_box = detect_face(img, filename)
            
            # Save visualization if enabled
            if VISUALIZE:
                save_debug_visualization(img, face_box, dst_filename)
            
            if face_box is not None:
                # Face detected, use subject-based cropping
                img = subject_based_crop(img, face_box)
                face_cropped += 1
                crop_method = "Face"
            else:
                # No face detected, fall back to central cropping
                img = central_square_crop(img)
                central_cropped += 1
                crop_method = "Central"
            
            # Resize to 128x128
            img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LANCZOS4)
            
            # Save the processed image
            cv2.imwrite(dst_path, img)
            
            image_counter += 1
            pbar.set_postfix({
                "Face crops": face_cropped, 
                "Central crops": central_cropped,
                "Face ratio": f"{face_cropped/max(1, image_counter):.1%}"
            })
            pbar.update(1)
                
        except Exception as e:
            pbar.write(f"Error processing {src_path}: {e}")
            if DEBUG:
                import traceback
                traceback.print_exc()

pbar.close()

print(f"\nAll images processed and saved to {REAL_SUBDIR}")
print(f"Total images processed: {image_counter}")
print(f"Images cropped using face detection: {face_cropped} ({face_cropped/max(1, image_counter):.1%})")
print(f"Images cropped using central cropping: {central_cropped} ({central_cropped/max(1, image_counter):.1%})")
print(f"Face detection method: Enhanced Dlib with multi-scale detection")

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