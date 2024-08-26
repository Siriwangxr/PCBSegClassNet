import os
import numpy as np
from sklearn.model_selection import train_test_split
import shutil

# Set directories
crops_dir = '/home/xinrui/projects/PCB_Segmentation/dataset/FPIC/segmentation/crops_2'
masks_dir = '/home/xinrui/projects/PCB_Segmentation/dataset/FPIC/segmentation/masks'

# Define categories to process
drop_categories = ['R', 'C']
all_categories = [cat for cat in os.listdir(crops_dir) if os.path.isdir(os.path.join(crops_dir, cat))]
keep_categories = [cat for cat in all_categories if cat not in drop_categories]


# Function to load file paths
def load_file_paths(category):
    crop_cat_dir = os.path.join(crops_dir, category)
    mask_cat_dir = os.path.join(masks_dir, category)

    crop_files = [os.path.join(crop_cat_dir, f) for f in os.listdir(crop_cat_dir) if
                  os.path.isfile(os.path.join(crop_cat_dir, f))]
    mask_files = [os.path.join(mask_cat_dir, f) for f in os.listdir(mask_cat_dir) if
                  os.path.isfile(os.path.join(mask_cat_dir, f))]

    return crop_files, mask_files


# Function to randomly drop 75% of the files
def drop_files(crop_files, mask_files, drop_percentage=0.75):
    crop_files = sorted(crop_files)
    mask_files = sorted(mask_files)
    total_files = len(crop_files)
    indices = np.arange(total_files)
    np.random.shuffle(indices)
    drop_indices = indices[:int(drop_percentage * total_files)]
    keep_indices = indices[int(drop_percentage * total_files):]

    keep_crop_files = sorted([crop_files[i] for i in keep_indices])
    keep_mask_files = sorted([mask_files[i] for i in keep_indices])

    return keep_crop_files, keep_mask_files


# Load and process drop categories
crops_files = []
masks_files = []

for category in drop_categories:
    crop_files, mask_files = load_file_paths(category)
    keep_crop_files, keep_mask_files = drop_files(crop_files, mask_files)
    crops_files.extend(keep_crop_files)
    masks_files.extend(keep_mask_files)

# Load and process keep categories
for category in keep_categories:
    crop_files, mask_files = load_file_paths(category)
    crop_files = sorted(crop_files)
    mask_files = sorted(mask_files)
    crops_files.extend(crop_files)
    masks_files.extend(mask_files)

crops_files = sorted(crops_files)
masks_files = sorted(masks_files)
for i in range(len(crops_files)):
    index_crops = crops_files[i].split('/')[-1].split('.')[0].split('_')[1]
    index_masks = masks_files[i].split('/')[-1].split('.')[0].split('_')[1]
    if index_crops != index_masks:
        print(index_crops, index_masks)



# Split remaining data into train, validation, and test sets
total_files = len(crops_files)
indices = np.arange(total_files)
train_indices, test_indices = train_test_split(indices, test_size=0.15, random_state=42)
train_indices, val_indices = train_test_split(train_indices, test_size=0.15, random_state=42)

# Create directories for the split datasets
split_dirs = ['train', 'val', 'test']
for split_dir in split_dirs:
    os.makedirs(os.path.join('/home/xinrui/projects/PCB_Segmentation/dataset/FPIC/ufl_dataset_2', split_dir, 'crops'), exist_ok=True)
    os.makedirs(os.path.join('/home/xinrui/projects/PCB_Segmentation/dataset/FPIC/ufl_dataset_2', split_dir, 'masks'), exist_ok=True)


# Function to copy files to the corresponding split directory
def copy_files(indices, split_name):
    for idx in indices:
        crop_file = crops_files[idx]
        mask_file = masks_files[idx]

        # Copy crop file
        crop_dest_dir = os.path.join('/home/xinrui/projects/PCB_Segmentation/dataset/FPIC/ufl_dataset_2', split_name, 'crops')
        os.makedirs(crop_dest_dir, exist_ok=True)
        shutil.copy(crop_file, os.path.join(crop_dest_dir, os.path.basename(crop_file)))

        # Copy mask file
        mask_dest_dir = os.path.join('/home/xinrui/projects/PCB_Segmentation/dataset/FPIC/ufl_dataset_2', split_name, 'masks')
        os.makedirs(mask_dest_dir, exist_ok=True)
        shutil.copy(mask_file, os.path.join(mask_dest_dir, os.path.basename(mask_file)))


# Copy files to their respective directories
copy_files(train_indices, 'train')
copy_files(val_indices, 'val')
copy_files(test_indices, 'test')

print("Dataset splitting completed successfully!")

