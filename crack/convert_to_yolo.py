import os
import cv2
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def process_dataset(images_dir, masks_dir, output_dir, split_ratio=0.8):
    # Setup output directories
    output_path = Path(output_dir)
    images_train = output_path / 'images' / 'train'
    images_val = output_path / 'images' / 'val'
    labels_train = output_path / 'labels' / 'train'
    labels_val = output_path / 'labels' / 'val'
    
    for p in [images_train, images_val, labels_train, labels_val]:
        p.mkdir(parents=True, exist_ok=True)
        
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
    print(f"Found {len(image_files)} images in dataset.")
    
    # Shuffle for random split
    random.shuffle(image_files)
    split_idx = int(len(image_files) * split_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"Splitting into {len(train_files)} training and {len(val_files)} validation samples.")
    
    def process_split(files, split_name):
        img_dest_dir = images_train if split_name == 'train' else images_val
        lbl_dest_dir = labels_train if split_name == 'train' else labels_val
        
        for filename in tqdm(files, desc=f"Processing {split_name} data"):
            img_path = os.path.join(images_dir, filename)
            mask_path = os.path.join(masks_dir, filename)
            
            if not os.path.exists(mask_path):
                print(f"Warning: Mask not found for {filename}")
                continue
                
            # Copy image
            shutil.copy(img_path, img_dest_dir / filename)
            
            # Read mask and get contours
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
                
            H, W = mask.shape
            # Assuming crack mask is white (255) on black (0) background
            # If it's black on white, use cv2.THRESH_BINARY_INV
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Write YOLO labels
            label_filename = filename.rsplit('.', 1)[0] + '.txt'
            label_path = lbl_dest_dir / label_filename
            
            with open(label_path, 'w') as f:
                for contour in contours:
                    # Filter out very small noise contours (e.g., area < 10)
                    if cv2.contourArea(contour) < 10:
                        continue
                        
                    # Flatten the contour array
                    contour = contour.flatten()
                    
                    if len(contour) < 6: # Need at least 3 points (x, y) for a polygon
                        continue
                        
                    # Normalize points to 0-1 range
                    normalized_points = []
                    for i in range(0, len(contour), 2):
                        x = contour[i] / W
                        y = contour[i+1] / H
                        # Ensure values are strictly within [0, 1]
                        x = max(0.0, min(1.0, x))
                        y = max(0.0, min(1.0, y))
                        normalized_points.extend([x, y])
                        
                    if len(normalized_points) > 0:
                        # Class ID is 0 for 'crack'
                        line = f"0 " + " ".join(f"{p:.6f}" for p in normalized_points) + "\n"
                        f.write(line)

    process_split(train_files, 'train')
    process_split(val_files, 'val')

if __name__ == "__main__":
    # Paths based on the discovered structure
    INPUT_IMAGES = r"d:\Crack\crack_dataset\Crack_Segmentation_Dataset\images"
    INPUT_MASKS = r"d:\Crack\crack_dataset\Crack_Segmentation_Dataset\masks"
    OUTPUT_DIR = r"d:\Crack\yolo_dataset"
    
    # Set random seed for reproducibility
    random.seed(42)
    
    print("Starting dataset conversion for YOLO segmentation...")
    process_dataset(INPUT_IMAGES, INPUT_MASKS, OUTPUT_DIR)
    print("Conversion complete!")
