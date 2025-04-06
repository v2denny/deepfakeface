'''
All images processed and saved to 30kds_real_face_crop_cv2\Real
Total images processed: 100
Images cropped using face detection: 82 (82.0%)
Images cropped using central cropping: 18 (18.0%)
Face detection method used: multi
'''


import os
import random
import cv2
import numpy as np
from tqdm import tqdm

# Global variables
LIMIT = 100  # Limit processing to specified number of images
DEBUG = True  # Enable debug mode with verbose output
VISUALIZE = True  # Save visualization of detected faces
FACE_DETECTION_METHOD = "multi"  # Options: "haar", "dnn", "multi"
CONF_THRESHOLD = 0.5  # Confidence threshold for DNN detection
IMAGE_LIMIT = LIMIT  # Using LIMIT for consistency

# Paths
INPUT_DIRS = ["30kds/train/Real", "30kds/val/Real", "30kds/test/Real"]
OUTPUT_DIR = "30kds_real_face_crop_cv2"
REAL_SUBDIR = os.path.join(OUTPUT_DIR, "Real")
DEBUG_DIR = os.path.join(OUTPUT_DIR, "Debug") if VISUALIZE else None

# Create output directories
os.makedirs(REAL_SUBDIR, exist_ok=True)
if VISUALIZE:
    os.makedirs(DEBUG_DIR, exist_ok=True)

# Load multiple face detection models for better coverage
print("Loading face detectors...")

# Haar Cascade face detectors (frontal and profile)
face_cascade_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

haar_loaded = not (face_cascade_frontal.empty() or face_cascade_profile.empty())
if haar_loaded:
    print("Haar Cascade face detectors loaded successfully")
else:
    print("Error: Could not load one or more Haar Cascade classifiers")

# Try to load DNN detector
dnn_loaded = False
try:
    # Use OpenCV's pre-trained face detection model
    net = cv2.dnn.readNetFromCaffe(
        "deploy.prototxt",
        "res10_300x300_ssd_iter_140000.caffemodel"
    )
    dnn_loaded = True
    print("DNN face detector loaded successfully")
except Exception as e:
    print(f"Warning: Could not load DNN face detector: {e}")
    print("You can download the models from:")
    print("https://github.com/opencv/opencv_extra/tree/master/testdata/dnn")

# Function for face detection using multiple methods
def detect_face(image, filename=""):
    height, width = image.shape[:2]
    faces = []
    detection_sources = []
    
    # Try DNN method first if available
    if dnn_loaded:
        # Create a blob from the image
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 
            1.0, 
            (300, 300), 
            (104.0, 177.0, 123.0)
        )
        
        # Pass the blob through the network
        net.setInput(blob)
        detections = net.forward()
        
        # Process detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filter based on confidence threshold
            if confidence > CONF_THRESHOLD:
                # Get the coordinates
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure coordinates are within image boundaries
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(width, endX)
                endY = min(height, endY)
                
                # Calculate width and height
                w = endX - startX
                h = endY - startY
                
                faces.append((startX, startY, w, h, confidence))
                detection_sources.append("DNN")
    
    # Use Haar Cascade methods if available
    if haar_loaded:
        # Convert to grayscale for Haar cascade
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Try different face detectors
        
        # Frontal face detector
        frontal_faces = face_cascade_frontal.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        for (x, y, w, h) in frontal_faces:
            # Assign a confidence of 0.9 for frontal faces (arbitrary high value)
            faces.append((x, y, w, h, 0.9))
            detection_sources.append("Haar-Frontal")
        
        # Profile face detector
        profile_faces = face_cascade_profile.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        for (x, y, w, h) in profile_faces:
            # Assign a confidence of 0.8 for profile faces (arbitrary high value)
            faces.append((x, y, w, h, 0.8))
            detection_sources.append("Haar-Profile")
        
        # Try flipped image for profile faces looking the other way
        flipped = cv2.flip(gray, 1)
        flipped_profile_faces = face_cascade_profile.detectMultiScale(
            flipped,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        for (x, y, w, h) in flipped_profile_faces:
            # Convert coordinates back to original image
            x = width - x - w
            # Assign a confidence of 0.7 for flipped profile faces
            faces.append((x, y, w, h, 0.7))
            detection_sources.append("Haar-Profile-Flipped")
    
    if DEBUG:
        print(f"\nDetection results for {filename}:")
        print(f"  Found {len(faces)} faces")
        for i, (x, y, w, h, conf) in enumerate(faces):
            print(f"  Face {i+1}: method={detection_sources[i]}, confidence={conf:.3f}, coords=[{x}, {y}, {w}, {h}]")
    
    # Find the best face (highest confidence, or largest area if same confidence)
    if faces:
        # First sort by confidence, then by area if confidence is the same
        best_face = max(faces, key=lambda f: (f[4], f[2] * f[3]))
        # Return only the box coordinates (x, y, w, h)
        return best_face[:4]
    
    if DEBUG:
        print("  No face detected")
    return None

# Function for central square cropping (fallback)
def central_square_crop(image):
    height, width = image.shape[:2]
    min_dim = min(width, height)
    
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    
    cropped = image[top:top + min_dim, left:left + min_dim]
    return cropped

# Function for subject-based cropping
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
            
            # Try to detect face
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
print(f"Face detection method used: {FACE_DETECTION_METHOD}")

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