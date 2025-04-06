# Dataset Creation Scripts

This directory contains scripts designed to preprocess images for training a deepfake generation model. During the generation phase, we observed that images often included excessive background noise, capturing more than just the faces. Simply resizing images was inadequate, as it led to the generator learning from distorted, stretched images. Central cropping was also insufficient, as it risked omitting facial features when faces were not centered. To address these issues, we developed methods to crop images more effectively, ensuring higher quality inputs for the generator and thereby improving the generation results.

## Cropping Success Rate Summary

| Method         | % Face Cropped | % Central Cropped | Detection Method |
|----------------|----------------|--------------------|------------------|
| OpenCV (cv2)   | 82.0%          | 18.0%              | multi            |
| MediaPipe      | 74.0%          | 26.0%              | static           |
| Dlib           | 89.0%          | 11.0%              | -                |
| Central Crop   | 0.0%           | 100.0%             | -                |

## Scripts Overview

### `create_30k_ds.py`
This is the main script that orchestrates the dataset creation process. It utilizes one of the face cropping methods (cv2, dlib, or mediapipe) to crop faces from images and compiles a dataset of 30,000 images.

### `create_30k_real_central_crop.py`
This script performs central cropping on images to create a dataset of 30,000 images. However, central cropping may inadvertently exclude important facial features if the face is not perfectly centered.

### `create_30k_real_face_crop_cv2.py`
Utilizes OpenCV (`cv2`) for face detection and cropping to generate a dataset of 30,000 images. OpenCV's face detection is generally fast but may not be as accurate as other methods, leading to a certain percentage of unsuccessful crops.

### `create_30k_real_face_crop_dlib.py` (chosen method)
Employs `dlib` for face detection and cropping to create a dataset of 30,000 images. While `dlib` offers accurate face detection, its approach takes longer processing time.

### `create_30k_real_face_crop_mediapipe.py`
Leverages Google's MediaPipe for face detection and cropping to assemble a dataset of 30,000 images. MediaPipe provides a balance between speed and accuracy, normally resulting in a higher percentage of successfully cropped images compared to some other methods.

---

By implementing these tailored cropping methods, we ensured that the images fed into the generator were of higher quality, focusing on the faces and minimizing background noise. This approach significantly enhanced the performance and realism of the generated deepfake outputs.
