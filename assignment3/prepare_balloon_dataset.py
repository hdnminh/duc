#!/usr/bin/env python
# coding: utf-8

"""
Download and prepare the Balloon dataset for Mask R-CNN training.
This script:
1. Downloads the Balloon dataset
2. Extracts it
3. Creates the necessary directory structure
4. Converts VIA annotations to mask images
5. Organizes files for processing
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import shutil
from tqdm import tqdm
import json
import cv2
import numpy as np


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def via_to_masks(via_annotations, image_dir, output_dir):
    """Convert VIA annotations to binary mask images"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load VIA annotations
    with open(via_annotations, 'r') as f:
        via_data = json.load(f)
    
    # Process each annotation
    for image_id, annotation in tqdm(via_data.items(), desc="Converting annotations to masks"):
        # Skip _via_settings_ entry if present
        if image_id == "_via_settings_":
            continue
            
        # Get image filename and load image to get dimensions
        filename = annotation["filename"]
        img_path = os.path.join(image_dir, filename)
        
        # Skip if image doesn't exist
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found. Skipping.")
            continue
            
        # Read image to get dimensions
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        
        # Create empty mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Get regions (polygons)
        regions = annotation["regions"]
        for i, region in enumerate(regions.values(), 1):  # Start from 1 for instance id
            # Get region shape and points
            shape_attributes = region["shape_attributes"]
            if shape_attributes["name"] != "polygon":
                print(f"Warning: Only polygon shapes are supported. Skipping {shape_attributes['name']} shape.")
                continue
            
            # Get polygon points
            x_points = shape_attributes["all_points_x"]
            y_points = shape_attributes["all_points_y"]
            points = list(zip(x_points, y_points))
            
            # Create a binary mask for this polygon
            polygon_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Convert points to numpy array for OpenCV
            points_array = np.array(points, dtype=np.int32)
            
            # Draw filled polygon
            cv2.fillPoly(polygon_mask, [points_array], 1)
            
            # Add to the main mask with different instance IDs
            mask[polygon_mask == 1] = i
        
        # Save mask image
        mask_filename = os.path.splitext(filename)[0] + ".png"
        mask_path = os.path.join(output_dir, mask_filename)
        cv2.imwrite(mask_path, mask)


def parse_args():
    parser = argparse.ArgumentParser(description='Download and prepare the Balloon dataset')
    parser.add_argument('--download_dir', type=str, default='.',
                        help='Directory to download and extract the dataset')
    parser.add_argument('--output_dir', type=str, default='balloon_dataset',
                        help='Output directory for prepared dataset')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create directories for images and masks
    images_dir = os.path.join(args.output_dir, 'images')
    masks_dir = os.path.join(args.output_dir, 'masks')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    # Download the dataset
    dataset_url = "https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip"
    zip_path = os.path.join(args.download_dir, "balloon_dataset.zip")
    
    if not os.path.exists(zip_path):
        print("Downloading Balloon dataset...")
        download_url(dataset_url, zip_path)
    
    # Extract the dataset
    extract_dir = os.path.join(args.download_dir, "balloon_extract")
    if not os.path.exists(extract_dir):
        print("Extracting Balloon dataset...")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    
    # Process train annotations
    train_dir = os.path.join(extract_dir, "balloon", "train")
    train_ann_file = os.path.join(train_dir, "via_region_data.json")
    
    # Convert train annotations to masks
    print("Converting training annotations to masks...")
    via_to_masks(train_ann_file, train_dir, masks_dir)
    
    # Process val annotations
    val_dir = os.path.join(extract_dir, "balloon", "val")
    val_ann_file = os.path.join(val_dir, "via_region_data.json")
    
    # Convert val annotations to masks
    print("Converting validation annotations to masks...")
    via_to_masks(val_ann_file, val_dir, masks_dir)
    
    # Copy all images to images directory
    print("Copying images...")
    for source_dir in [train_dir, val_dir]:
        for file in os.listdir(source_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')) and not file.startswith('via_'):
                shutil.copy(
                    os.path.join(source_dir, file),
                    os.path.join(images_dir, file)
                )
    
    # Create simple class info file
    class_info = {
        "categories": [
            {
                "id": 1,
                "name": "balloon",
                "supercategory": "none"
            }
        ]
    }
    
    with open(os.path.join(args.output_dir, "class_info.json"), 'w') as f:
        json.dump(class_info, f, indent=2)
    
    print(f"\nDataset preparation complete!")
    print(f"Dataset directory: {args.output_dir}")
    print(f"Number of images: {len(os.listdir(images_dir))}")
    print(f"Number of masks: {len(os.listdir(masks_dir))}")
    print("\nNext steps:")
    print("1. Convert to COCO format using mask_rcnn_segmentation.py --task convert")
    print("2. Train the model using mask_rcnn_segmentation.py --task train")


if __name__ == "__main__":
    main() 