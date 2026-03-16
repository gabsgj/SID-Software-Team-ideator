# Crack Detection & Segmentation Project Documentation

## 1. Project Overview
This project focuses on identifying and segmenting cracks in images using the ultralytics YOLOv11 framework. The goal is to produce a lightweight AI model (`yolo11n-seg.pt`) capable of running on edge devices such as drones.

The primary task handled in this setup phase is converting an existing dataset of images and pixel-level binary masks into the specialized polygon-contour format required by YOLO for instance segmentation.

## 2. Dataset Information
The raw dataset originates from a crack detection and segmentation repository designed for UAV (drone) inspection. It consists of:
*   Over **11,000** raw crack images.
*   Corresponding **binary masks** where cracked regions are highlighted in white against a black background.

## 3. The Conversion Process
YOLO segmentation models require coordinate polygons (a sequence of normalized $x,y$ points bounding the object), rather than pixel-level binary masks. 

### `convert_to_yolo.py` Script Details
We created a Python script that automates the following steps:
1.  **Reading Images & Masks:** It scans the source directory `Crack_Segmentation_Dataset/images` and `Crack_Segmentation_Dataset/masks` to find matching pairs.
2.  **Dataset Splitting:** The images are randomly shuffled and split. We chose a standard **80% Training / 20% Validation** split.
3.  **Contour Extraction:** Using `OpenCV` (`cv2.findContours`), the script analyzes the binary masks, extracting the outer boundary contours of every distinct crack.
4.  **Formatting and Normalizing:** The pixel coordinates of these contours are normalized (divided by image width and height to fit between 0 and 1). These points are then written into a `.txt` file along with the class ID (`0` for "crack").
5.  **Noise Filtering:** Any contours possessing an artificially small area (less than 10 pixels) are classified as noise and ignored.

### Output Structure
The conversion script outputs the data identically to YOLO requirements in a directory called `yolo_dataset`:
```
yolo_dataset/
├── images/
│   ├── train/  (6,045 images)
│   └── val/    (1,512 images)
└── labels/
    ├── train/  (6,045 text files)
    └── val/    (1,512 text files)
```

## 4. Configuration
We use a YAML configuration file (`crack_dataset.yaml`) to steer the YOLO training process. It explicitly points to the generated `yolo_dataset` folders and maps Class ID `0` to the noun "crack". 

## 5. Training Methodology
Initially, we started a YOLOv11 nano (`yolo11n-seg.pt`) training locally on your Windows machine's CPU (AMD Ryzen 5). While the YOLO CLI correctly loaded the dataset and started mapping the configuration, the estimated time per epoch was extremely high (~1 hour and 11 minutes per epoch, totaling ~30 hours for 25 epochs).

### Cloud Training Migration (Google Colab)
Running the model training on a cloud-based GPU (like an NVIDIA T4 on Google Colab) drastically cuts down training time from days to just a few minutes or hours depending on the epochs.

A robust secondary guide named **`Colab_Training_Guide.md`** has been provided. It details how to upload the ready-made `crack_yolo_data.zip` direct to Google Drive and execute the exact training blocks needed.
