# YOLOv11 Crack Segmentation: Google Colab Guide

Since YOLOv11 training requires significant compute, running it on Google Colab's free GPUs (like the T4) is highly recommended. This guide will walk you through moving the converted dataset to Colab and running the training.

## Step 1: Prepare the Dataset for Colab
Google Colab transfers large numbers of small files very slowly. It's much faster to upload a single ZIP file.

1. On your Windows machine, compress the converted `yolo_dataset` folder and the `crack_dataset.yaml` file into a single ZIP file (e.g., `crack_yolo_data.zip`).
   - *If you want to do this via PowerShell:*
     ```powershell
     Compress-Archive -Path d:\Crack\yolo_dataset, d:\Crack\crack_dataset.yaml -DestinationPath d:\Crack\crack_yolo_data.zip
     ```
2. Upload this `crack_yolo_data.zip` file to your Google Drive (e.g., place it in the root or a new folder called `YOLO_Crack_Training`).

## Step 2: Set up the Colab Notebook
1. Go to [Google Colab](https://colab.research.google.com/) and create a "New Notebook".
2. **⚠️ CRITICAL: Enable GPU:** Go to the top menu, click **`Runtime` > `Change runtime type`**.
3. Under **Hardware accelerator**, select **`T4 GPU`** and click Save. 
*(If you see an error like "Invalid CUDA 'device=0'", it means you forgot this step and are running on a CPU!)*

## Step 3: Mount Google Drive
In your first Colab cell, connect your Google Drive to access the dataset:

```python
from google.colab import drive
drive.mount('/content/drive')
```
*(Run the cell and follow the authentication prompts).*

## Step 4: Unzip the Dataset
In the next cell, copy the ZIP file from your Drive and extract it:

```bash
# Update the path below if you placed the zip in a different folder in Drive
!cp "/content/drive/MyDrive/crack_yolo_data.zip" /content/
!unzip -q /content/crack_yolo_data.zip -d /content/crack_data/
```

## Step 5: Install Ultralytics
Install the YOLO framework:

```bash
!pip install ultralytics
```

## Step 6: Update the YAML path for Colab
The `crack_dataset.yaml` you uploaded has Windows paths (`d:/Crack/...`). You need to update them to point to the Colab paths. Run this Python script in a cell to rewrite the YAML:

```python
import yaml

yaml_path = '/content/crack_data/crack_dataset.yaml'

with open(yaml_path, 'r') as file:
    data = yaml.safe_load(file)

# Update paths to the extracted Colab directory
data['path'] = '/content/crack_data/yolo_dataset'
data['train'] = 'images/train'
data['val'] = 'images/val'

with open(yaml_path, 'w') as file:
    yaml.dump(data, file)
    
print("Updated YAML paths for Colab!")
```

## Step 7: Train the Model!
Now start the training process. The T4 GPU will train the model *much* faster than a local CPU!

```python
from ultralytics import YOLO

# Load the nano segmentation model
model = YOLO('yolo11n-seg.pt')

# Train it! (Adjust epochs as needed, 25 is a good start)
results = model.train(
    data='/content/crack_data/crack_dataset.yaml',
    epochs=25,
    imgsz=640,
    batch=16,
    device=0  # 0 indicates the first GPU. If you MUST use a CPU, change this to device='cpu'
)
```

## Step 8: Save Your Results
When training finishes, Colab will save the weights in a `runs` folder. Because Colab instances reset when you close them, you MUST copy the final weights back to your Google Drive!

```bash
# Create a folder in your Drive to save the model
!mkdir -p "/content/drive/MyDrive/YOLO_Crack_Training/results"

# Copy the best weights and the training graphs
!cp /content/runs/segment/train/weights/best.pt "/content/drive/MyDrive/YOLO_Crack_Training/results/"
!cp /content/runs/segment/train/results.png "/content/drive/MyDrive/YOLO_Crack_Training/results/"

print("Model saved to your Google Drive!")
```
