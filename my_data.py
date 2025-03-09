import os
import shutil
import random
from pathlib import Path

def my_data(src_root, dst_root, val_ratio=0.1, seed=42):
    """
    Splits an existing dataset (train & test folders) into train, validation, and test.
    
    Parameters:
      src_root (Path): Source directory containing 'train' and 'test' folders.
      dst_root (Path): Destination directory where new split folders will be created.
      val_ratio (float): Fraction of training data to use for validation.
      seed (int): Random seed for reproducibility.
    """
    random.seed(seed)
    
    # Define source paths
    train_src = Path(src_root) / "train"
    test_src = Path(src_root) / "test"
    
    # Define destination paths
    train_dst = Path(dst_root) / "train"
    val_dst = Path(dst_root) / "val"
    test_dst = Path(dst_root) / "test"
    
    # Ensure all destination directories exist
    for path in [train_dst, val_dst, test_dst]:
        path.mkdir(parents=True, exist_ok=True)
    
    # Process each class in the training set
    for cls in os.listdir(train_src):
        class_train_src = train_src / cls
        class_train_dst = train_dst / cls
        class_val_dst = val_dst / cls
        
        class_train_dst.mkdir(parents=True, exist_ok=True)
        class_val_dst.mkdir(parents=True, exist_ok=True)
        
        images = [p for p in class_train_src.iterdir() if p.is_file()]
        random.shuffle(images)
        
        n_val = int(len(images) * val_ratio)
        train_imgs, val_imgs = images[n_val:], images[:n_val]
        
        print(f"Class: {cls} | Train: {len(train_imgs)} | Val: {len(val_imgs)}")
        
        for file_path in train_imgs:
            shutil.copy(file_path, class_train_dst / file_path.name)
        for file_path in val_imgs:
            shutil.copy(file_path, class_val_dst / file_path.name)
    
    # Copy the test set as is
    for cls in os.listdir(test_src):
        class_test_src = test_src / cls
        class_test_dst = test_dst / cls
        
        class_test_dst.mkdir(parents=True, exist_ok=True)
        
        for file_path in class_test_src.iterdir():
            if file_path.is_file():
                shutil.copy(file_path, class_test_dst / file_path.name)
    
    print("Dataset processing completed!")

# Example usage
if __name__ == '__main__':
    src_path = Path("/kaggle/input/my-dataset")  # Change this to your dataset path
    dst_path = Path("/kaggle/working/my-dataset-split")
    
    my_data(src_path, dst_path, val_ratio=0.1)
    print("Splitting done!")
