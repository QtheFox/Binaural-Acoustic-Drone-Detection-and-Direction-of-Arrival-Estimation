import os
import glob
import shutil
import random

# -----------------------------
# Parameters
# -----------------------------
dataset_root = "dataset_real2"           # Original dataset folder
train_root = "dataset_training_real2"   # Training split folder
val_root = "dataset_validation_real2"   # Validation split folder
#dataset_root = "dataset"           # Original dataset folder
#train_root = "dataset_training"   # Training split folder
#val_root = "dataset_validation"   # Validation split folder
train_ratio = 0.8                 # 80% training, 20% validation
random_seed = 42

random.seed(random_seed)

# -----------------------------
# Create output folders
# -----------------------------
os.makedirs(train_root, exist_ok=True)
os.makedirs(val_root, exist_ok=True)

# -----------------------------
# Process each azimuth folder
# -----------------------------
az_folders = sorted(glob.glob(os.path.join(dataset_root, 'Az*_El*')))
for az_folder in az_folders:
    az_name = os.path.basename(az_folder)
    snippet_folders = sorted(glob.glob(os.path.join(az_folder, '*')))
    
    random.shuffle(snippet_folders)  # Shuffle snippets within this azimuth

    split_idx = int(len(snippet_folders) * train_ratio)
    train_snippets = snippet_folders[:split_idx]
    val_snippets = snippet_folders[split_idx:]

    # Copy to training folder
    for snippet in train_snippets:
        dest_folder = os.path.join(train_root, az_name, os.path.basename(snippet))
        os.makedirs(os.path.dirname(dest_folder), exist_ok=True)
        shutil.copytree(snippet, dest_folder)

    # Copy to validation folder
    for snippet in val_snippets:
        dest_folder = os.path.join(val_root, az_name, os.path.basename(snippet))
        os.makedirs(os.path.dirname(dest_folder), exist_ok=True)
        shutil.copytree(snippet, dest_folder)

print("Dataset split complete!")