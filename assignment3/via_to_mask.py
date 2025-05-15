#!/usr/bin/env python
# coding: utf-8

"""
Convert annotations from VIA format (JSON) to binary mask images.
Specifically designed for the Balloon dataset, but can be adapted for other datasets.
"""

import os
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Convert VIA annotations to mask images')
    parser.add_argument('--annotations', type=str, required=True,
                        help='Path to the VIA annotations JSON file')
    parser.add_argument('--images', type=str, required=True,
                        help='Directory containing the images')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for mask images')
    return parser.parse_args()


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


def main():
    args = parse_args()
    via_to_masks(args.annotations, args.images, args.output)
    print(f"Conversion complete. Mask images saved to {args.output}")


if __name__ == "__main__":
    main() 