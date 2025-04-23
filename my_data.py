import os
import shutil
import random
from pathlib import Path

def create_directory_structure(destination_root, classes):
    """
    Create the necessary directory structure for the dataset.
    
    Args:
        destination_root (str): Root directory where the dataset will be stored.
        classes (list): List of class names.
    """
    for split in ['train', 'validation', 'test']:
        for class_name in classes:
            os.makedirs(os.path.join(destination_root, split, class_name), exist_ok=True)
            
def move_files(source_dir, dest_dir, ratio, random_seed=42):
    """
    Move a random subset of files from source_dir to dest_dir.
    
    Args:
        source_dir (str): Source directory containing the files.
        dest_dir (str): Destination directory where files will be moved.
        ratio (float): Ratio of files to move (0-1).
        random_seed (int): Random seed for reproducibility.
    
    Returns:
        int: Number of files moved.
    """
    random.seed(random_seed)
    
    # Get all files in the source directory
    files = os.listdir(source_dir)
    
    # Calculate number of files to move
    num_files_to_move = int(len(files) * ratio)
    
    # Randomly select files to move
    files_to_move = random.sample(files, num_files_to_move)
    
    # Move files
    for file in files_to_move:
        source_path = os.path.join(source_dir, file)
        dest_path = os.path.join(dest_dir, file)
        shutil.move(source_path, dest_path)
    
    return num_files_to_move

def copy_dataset(source_root, destination_root):
    """
    Copy the dataset from source_root to destination_root preserving the structure.
    
    Args:
        source_root (str): Root directory of the original dataset.
        destination_root (str): Root directory where the dataset will be copied.
        
    Returns:
        list: List of class names.
    """
    # Get class names from train directory
    classes = os.listdir(os.path.join(source_root, 'train'))
    
    # Create destination directory structure
    create_directory_structure(destination_root, classes)
    
    # Copy files from source to destination
    for split in ['train', 'test']:
        for class_name in classes:
            source_dir = os.path.join(source_root, split, class_name)
            dest_dir = os.path.join(destination_root, split, class_name)
            
            # Get all files
            files = os.listdir(source_dir)
            
            # Copy files
            for file in files:
                source_path = os.path.join(source_dir, file)
                dest_path = os.path.join(dest_dir, file)
                shutil.copy2(source_path, dest_path)
    
    return classes

def reorganize_dataset(source_root, destination_root, source_split='test', target_split='train', move_ratio=0.1, random_seed=42):
    """
    Copy the dataset and reorganize by moving files between splits.
    
    Args:
        source_root (str): Root directory of the original dataset.
        destination_root (str): Root directory where the reorganized dataset will be stored.
        source_split (str): Source folder for moving files ('train' or 'test').
        target_split (str): Target folder to move files to ('train', 'validation', or 'test').
        move_ratio (float): Ratio of files to move (0-1).
        random_seed (int): Random seed for reproducibility.
        
    Returns:
        int: Total number of files moved.
    """
    # Copy dataset and get class names
    classes = copy_dataset(source_root, destination_root)
    
    # Move files from source_split to target_split
    files_moved = 0
    for class_name in classes:
        source_dir = os.path.join(destination_root, source_split, class_name)
        dest_dir = os.path.join(destination_root, target_split, class_name)
        
        files_moved += move_files(source_dir, dest_dir, move_ratio, random_seed)
    
    return files_moved

def count_images(dataset_root):
    """
    Count and print the number of images in each split and class.
    
    Args:
        dataset_root (str): Root directory of the dataset.
    """
    splits = ['train', 'validation', 'test']
    
    # Get class names from train directory
    try:
        classes = os.listdir(os.path.join(dataset_root, 'train'))
    except FileNotFoundError:
        print(f"Directory {os.path.join(dataset_root, 'train')} not found.")
        return
    
    # Count images in each split and class
    for split in splits:
        print(f"{split} set:")
        split_total = 0
        
        for class_name in classes:
            dir_path = os.path.join(dataset_root, split, class_name)
            
            if os.path.exists(dir_path):
                num_files = len(os.listdir(dir_path))
                print(f"  {class_name}: {num_files} images")
                split_total += num_files
            else:
                print(f"  {class_name}: 0 images (directory not found)")
        
        print(f"  Total: {split_total} images")
        print()

def main():
    """
    Main function to execute the dataset reorganization.
    """
    # Define paths
    source_root = '/kaggle/input/labeled-chest-xray-images/chest_xray'
    destination_root = '/kaggle/working/labeled-chest-xray-images'
    
    # Set parameters - modify these to change the source and target of the split
    source_split = 'test'    # Source for moving files ('train' or 'test')
    target_split = 'train'   # Target to move files to ('train', 'validation', or 'test')
    move_ratio = 0.1         # Ratio of files to move
    random_seed = 42         # Random seed for reproducibility
    
    # To move files from train to validation instead, change these parameters:
    # source_split = 'train'
    # target_split = 'validation'
    
    # Reorganize dataset
    files_moved = reorganize_dataset(
        source_root, 
        destination_root, 
        source_split, 
        target_split, 
        move_ratio, 
        random_seed
    )
    
    print(f"Moved {files_moved} files from {source_split} to {target_split}")
    print()
    
    # Print statistics
    print("Dataset statistics:")
    count_images(destination_root)

if __name__ == "__main__":
    main()