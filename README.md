# DeepFakeFace

This repository contains the implementation of a deep learning project focused on building both **discriminative** and **generative** models in the context of deepfakes. The goal is to classify real vs. fake faces and to generate realistic fake face images using various architectures and data preprocessing techniques.


## Objective

- **Discriminative Task**: Classify images as real or fake.
- **Generative Task**: Generate synthetic face images that resemble real ones.

The models are trained and evaluated on the **DeepFakeFace (DFF)** dataset, which includes:
- 30,000 real images from IMDB-WIKI.
- 90,000 fake images generated using:
  - Stable Diffusion v1.5
  - Stable Diffusion Inpainting
  - InsightFace


## Approach

- **Data Preprocessing**: For the classification task, no major preprocessing was done, the images were just resized accordingly to the model used. For the generation task, images were preprocessed using different cropping strategies to reduce background noise and focus on faces. Central cropping alone was insufficient, so we used face detection with OpenCV and MediaPipe to get better training inputs. Dlib was also tested and ended up being the method used even though its slow iteration.

- **Data Distribution**: For classification: 30k real images and 30k fake images (10k from each dataset).
For generation: all the 30k real images.

- **Classification**:
  - Models: CNN, Vision Transformer (ViT), and YOLO.
  - Evaluation: Accuracy, F1-score, precision, recall.

- **Generation**:
  - Models: dcGAN, StyleGAN.
  - Evaluation: Visual quality, FID, LPIPS.


## Project Structure

```
.
├── dataset_creation/               # Scripts for face cropping and dataset creation
│   ├── create_30k_ds.py
│   ├── create_30k_real_central_crop.py
│   ├── create_30k_real_face_crop_cv2.py
│   ├── create_30k_real_face_crop_dlib.py
│   └── create_30k_real_face_crop_mediapipe.py
│
├── models/                         # Saved model architectures and checkpoints
│
├── classifier_CNN.ipynb            # CNN classifier training notebook
├── classifier_ViT.ipynb            # Vision Transformer classifier
├── classifier_yolo.ipynb           # YOLO-based classifier
├── classifier_benchmarking.ipynb   # Comparing classifier performances
├── generator_dcGAN.ipynb           # GAN training (dcGAN)
├── generator_styleGAN.ipynb        # GAN training (StyleGAN)
├── eda.ipynb                        # Exploratory Data Analysis
├── README.md
└── requirements.txt                # Environment dependencies
```


## Results & Evaluation

- **CNN**: Baseline classifier with strong performance.
- **ViT**: Improved accuracy on complex fake images.
- **YOLO**: Tested for detection + classification tasks.
- **dcGAN vs. StyleGAN**: StyleGAN produced more realistic faces.

The results are explored in depth both in the `classifier_benchmarking.ipynb` and `aaa` files as well as on the report.

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run dataset preparation:
```bash
python dataset_creation/create_30k_real_face_crop_cv2.py
```

3. Launch classification training:
```bash
jupyter notebook classifier_CNN.ipynb
```

4. Launch generation training:
```bash
jupyter notebook generator_dcGAN.ipynb
```


## Notes
- Dataset must be manually downloaded from [HuggingFace - DeepFakeFace](https://huggingface.co/datasets/OpenRL/DeepFakeFace)
- The project emphasizes iterative improvement based on intermediate results, specifically the improved cropping technique.


## Authors
Project developed for the Deep and Reinforcement Learning (M.IA003) course, FEUP/FCUP 2024/2025.
Daniel, Rafael, Lucas.

