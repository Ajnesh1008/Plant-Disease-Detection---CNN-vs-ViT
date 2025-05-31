# Training script for both models
import os
import shutil
import random
from glob import glob

# Configuration
#original_dataset = 'data/raw_dataset/PlantVillage'  # e.g. unzipped PlantVillage folder
original_dataset = 'data/raw_dataset'  # No /PlantVillage

output_base = 'data/plant_disease_dataset'
train_ratio = 0.8

# Create folders
os.makedirs(f'{output_base}/train', exist_ok=True)
os.makedirs(f'{output_base}/val', exist_ok=True)

# Discover class folders
class_folders = [f for f in os.listdir(original_dataset) if os.path.isdir(os.path.join(original_dataset, f))]

for class_name in class_folders:
    img_paths = glob(os.path.join(original_dataset, class_name, '*.jpg'))
    random.shuffle(img_paths)
    split_idx = int(len(img_paths) * train_ratio)
    train_imgs, val_imgs = img_paths[:split_idx], img_paths[split_idx:]

    train_class_dir = os.path.join(output_base, 'train', class_name)
    val_class_dir = os.path.join(output_base, 'val', class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)

    for img in train_imgs:
        shutil.copy(img, train_class_dir)

    for img in val_imgs:
        shutil.copy(img, val_class_dir)

    print(f"Class '{class_name}' | Train: {len(train_imgs)} | Val: {len(val_imgs)}")
