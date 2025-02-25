import os
import shutil
import random
from pathlib import Path

def split_dataset(src_root, dst_root, classes, split_ratios, seed=42):
    """
    Splits dataset from src_root into dst_root using given split_ratios.
    
    Parameters:
      src_root (Path): Source directory containing class subfolders.
      dst_root (Path): Destination directory where split folders (train, val, test) will be created.
      classes (list): List of class names (subdirectory names) present in src_root.
      split_ratios (dict): Dictionary with keys 'train', 'val', 'test' and their corresponding ratios.
      seed (int): Random seed for shuffling.
    """
    # Set the random seed for reproducibility
    random.seed(seed)
    
    # Create destination directory structure: dst_root/train, dst_root/val, dst_root/test,
    # each containing subdirectories for each class.
    for split in split_ratios.keys():
        for cls in classes:
            split_class_dir = dst_root / split / cls
            split_class_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each class separately
    for cls in classes:
        class_src_dir = src_root / cls
        # List all files in the class directory. Here we assume images have an extension with a dot.
        images = [p for p in class_src_dir.iterdir() if p.is_file() and p.suffix]
        random.shuffle(images)
        
        total = len(images)
        n_train = int(total * split_ratios['train'])
        n_val = int(total * split_ratios['val'])
        # Ensure all images are used (any rounding differences go to the test set)
        n_test = total - n_train - n_val
        
        # Partition the list of images
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train+n_val]
        test_imgs = images[n_train+n_val:]
        
        print(f"Class: {cls} | Total: {total} | Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")
        
        # Define a helper function for copying files
        def copy_files(file_list, split):
            for file_path in file_list:
                dest_file = dst_root / split / cls / file_path.name
                # Copy the file. If you want to move instead of copy, use shutil.move(file_path, dest_file)
                shutil.copy(file_path, dest_file)
        
        copy_files(train_imgs, 'train')
        copy_files(val_imgs, 'val')
        copy_files(test_imgs, 'test')

if __name__ == '__main__':
    # Define source and destination paths using pathlib for ease of use.
    src_path = Path('/kaggle/input/covid-cxr-image-dataset-research/COVID_IEEE')
    dst_path = Path('/kaggle/working/COVID_IEEE')
    
    # Define the classes (subdirectories in the source dataset)
    class_names = ['covid', 'normal', 'virus']
    
    # Define the split ratios (they should add up to 1.0)
    ratios = {
        'train': 0.7,
        'val': 0.1,
        'test': 0.2
    }
    
    # Run the split
    split_dataset(src_path, dst_path, class_names, ratios)
    
    print("Dataset split completed successfully!")
