import os
import csv
import argparse
from pathlib import Path
import glob

def generate_labels_and_dataset(root_dir, dataset_type, labels_file, dataset_file):
    """
    Generates labels.txt and dataset.csv for the specified dataset type.

    Parameters:
    - root_dir (str): Root directory of the imagenette2-320 dataset.
    - dataset_type (str): 'train' or 'val' to specify the dataset split.
    - labels_file (str): Path to save labels.txt.
    - dataset_file (str): Path to save dataset.csv.
    """
    dataset_dir = Path(root_dir) / dataset_type
    if not dataset_dir.exists():
        print(f"Error: The directory {dataset_dir} does not exist.")
        return

    # Get all class directories
    class_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    # print(class_dirs)
    class_dirs_sorted = sorted(class_dirs, key=lambda x: x.name)  # Sort for consistency

    # Map class names to indices
    class_to_idx = {cls_dir.name: idx for idx, cls_dir in enumerate(class_dirs_sorted)}

    # Write labels.txt
    with open(labels_file, 'w') as lf:
        for cls_dir in class_dirs_sorted:
            lf.write(f"{cls_dir.name}\n")
    print(f"labels.txt generated with {len(class_to_idx)} classes.")

    # Prepare dataset.csv
    dataset_entries = []
    for cls_dir in class_dirs_sorted:
        class_name = cls_dir.name
        class_idx = class_to_idx[class_name]
        # Iterate over all JPEG images in the class directory
        for img_file in cls_dir.glob('*.JPEG'):
            img_path = img_file.resolve()
            dataset_entries.append([str(img_path), class_idx])

    # Write dataset.csv
    with open(dataset_file, 'w', newline='') as df:
        writer = csv.writer(df)
        writer.writerow(['image_path', 'label'])  # Header
        writer.writerows(dataset_entries)
    print(f"dataset.csv generated with {len(dataset_entries)} entries.")

def main():
    parser = argparse.ArgumentParser(description="Generate labels.txt and dataset.csv for imagenette2-320 dataset.")
    parser.add_argument('--root_dir', type=str, required=True, help="Root directory of the imagenette2-320 dataset.")
    parser.add_argument('--dataset_type', type=str, choices=['train', 'val'], required=True, help="Dataset split to process: 'train' or 'val'.")
    parser.add_argument('--labels_file', type=str, default='labels.txt', help="Output path for labels.txt.")
    parser.add_argument('--dataset_file', type=str, default='dataset.csv', help="Output path for dataset.csv.")

    args = parser.parse_args()

    generate_labels_and_dataset(args.root_dir, args.dataset_type, args.labels_file, args.dataset_file)

if __name__ == "__main__":
    main()