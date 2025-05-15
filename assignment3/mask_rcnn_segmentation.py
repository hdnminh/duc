#!/usr/bin/env python
# coding: utf-8

"""
Mask R-CNN Segmentation with MMDetection
This script demonstrates:
- Inference with pre-trained Mask R-CNN model
- Converting custom dataset to COCO format
- Fine-tuning the model
- Modifying model configuration
- Training with logging
- Visualization with TensorBoard
- Evaluation
"""

import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import argparse
import shutil
from pycocotools.coco import COCO
from mmdet.apis import init_detector, inference_detector
from mmdet.apis import train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
from mmdet.utils.analysis import get_model_complexity_info
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmcv.cnn import fuse_conv_bn
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Mask R-CNN for instance segmentation')
    parser.add_argument('--task', choices=['inference', 'convert', 'train', 'evaluate'], 
                        default='inference', help='Task to perform')
    parser.add_argument('--config', type=str, 
                        default='configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py',
                        help='Path to the model config file')
    parser.add_argument('--checkpoint', type=str, 
                        default=None,
                        help='Path to the checkpoint file')
    parser.add_argument('--input', type=str, default=None,
                        help='Input for inference or dataset path for training')
    parser.add_argument('--output', type=str, default='output',
                        help='Output directory')
    parser.add_argument('--score_threshold', type=float, default=0.3,
                        help='Score threshold for inference')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use, cuda:0 or cpu')
    parser.add_argument('--custom_dataset', type=str, default=None,
                        help='Path to custom dataset for conversion')
    parser.add_argument('--custom_dataset_output', type=str, default=None,
                        help='Path to save the converted dataset')
    parser.add_argument('--epochs', type=int, default=12,
                        help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for training')
    parser.add_argument('--workers', type=int, default=2,
                        help='Number of workers for dataloader')
    
    return parser.parse_args()


def visualize_result(img, result, score_threshold=0.3, save_path=None, class_names=None):
    """Visualize the detection results on the image."""
    if isinstance(img, str):
        img = cv2.imread(img)
    img = img.copy()
    
    if class_names is None:
        # Use COCO class names by default
        class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    # If result is a tuple, it contains bboxes and masks
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    
    # Draw bounding boxes
    colors = plt.cm.get_cmap('tab20', len(class_names)).colors[:, :3]
    colors = (colors * 255).astype(np.uint8).tolist()
    
    for i, bboxes in enumerate(bbox_result):
        if len(bboxes) == 0:
            continue
        
        class_name = class_names[i]
        color = colors[i % len(colors)]
        
        for bbox in bboxes:
            x1, y1, x2, y2, score = bbox
            if score < score_threshold:
                continue
                
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            text = f'{class_name}: {score:.2f}'
            cv2.putText(img, text, (int(x1), int(y1 - 5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw segmentation masks
    if segm_result is not None:
        for i, masks in enumerate(segm_result):
            if len(masks) == 0:
                continue
                
            color = colors[i % len(colors)]
            
            for j, mask in enumerate(masks):
                if bbox_result[i][j][4] < score_threshold:
                    continue
                    
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(img, contours, -1, color, 2)
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img)
    
    return img


def perform_inference(model, img_path, output_dir, score_threshold=0.3):
    """Perform inference with a pre-trained model"""
    os.makedirs(output_dir, exist_ok=True)
    
    # If img_path is a directory, process all images
    if os.path.isdir(img_path):
        image_files = [os.path.join(img_path, f) for f in os.listdir(img_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        image_files = [img_path]
    
    for img_file in tqdm(image_files, desc="Processing images"):
        # Read the image
        img = cv2.imread(img_file)
        if img is None:
            print(f"Failed to read image: {img_file}")
            continue
        
        # Perform inference
        result = inference_detector(model, img)
        
        # Save and display result
        output_file = os.path.join(output_dir, os.path.basename(img_file))
        visualize_result(img, result, score_threshold, output_file)
        
        print(f"Saved result to {output_file}")


def convert_to_coco_format(dataset_dir, output_dir):
    """Convert a custom dataset to COCO format for instance segmentation"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create directories for images
    train_img_dir = os.path.join(output_dir, 'train')
    val_img_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    
    # Read annotations if they exist
    ann_file = os.path.join(dataset_dir, 'annotations.json')
    if os.path.exists(ann_file):
        with open(ann_file, 'r') as f:
            annotations = json.load(f)
    else:
        # Create empty COCO annotations structure
        annotations = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # Define your categories here
        # For example, for balloon dataset
        annotations['categories'].append({
            'id': 1,
            'name': 'balloon',
            'supercategory': 'none'
        })
    
    # Split data into train and validation sets
    images = os.listdir(os.path.join(dataset_dir, 'images'))
    img_files = [f for f in images if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Use 80% for training, 20% for validation
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(img_files)
    split_idx = int(len(img_files) * 0.8)
    train_files = img_files[:split_idx]
    val_files = img_files[split_idx:]
    
    # Process training images
    train_coco = {'images': [], 'annotations': [], 'categories': annotations['categories']}
    process_dataset_split(dataset_dir, train_img_dir, train_files, train_coco)
    
    # Process validation images
    val_coco = {'images': [], 'annotations': [], 'categories': annotations['categories']}
    process_dataset_split(dataset_dir, val_img_dir, val_files, val_coco)
    
    # Save COCO annotation files
    with open(os.path.join(output_dir, 'train.json'), 'w') as f:
        json.dump(train_coco, f)
    
    with open(os.path.join(output_dir, 'val.json'), 'w') as f:
        json.dump(val_coco, f)
    
    print(f"Converted dataset saved to {output_dir}")
    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")


def process_dataset_split(dataset_dir, output_img_dir, img_files, coco_dict):
    """Process a dataset split (train/val) and update COCO dictionary"""
    annotation_id = 1
    
    for img_id, img_file in enumerate(img_files, 1):
        # Copy image to output directory
        src_path = os.path.join(dataset_dir, 'images', img_file)
        dst_path = os.path.join(output_img_dir, img_file)
        shutil.copy(src_path, dst_path)
        
        # Read image to get dimensions
        img = cv2.imread(src_path)
        height, width = img.shape[:2]
        
        # Add image info
        coco_dict['images'].append({
            'id': img_id,
            'file_name': img_file,
            'height': height,
            'width': width
        })
        
        # Process annotations if they exist
        mask_file = os.path.join(dataset_dir, 'masks', os.path.splitext(img_file)[0] + '.png')
        if os.path.exists(mask_file):
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            
            # Process each instance in the mask
            instance_ids = np.unique(mask)
            for instance_id in instance_ids:
                if instance_id == 0:  # Background
                    continue
                
                # Create binary mask for this instance
                binary_mask = (mask == instance_id).astype(np.uint8)
                
                # Find contours
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Create segmentation
                segmentation = []
                for contour in contours:
                    contour = contour.flatten().tolist()
                    if len(contour) > 4:  # Polygon must have at least 3 points
                        segmentation.append(contour)
                
                if not segmentation:
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(binary_mask)
                
                # Get area
                area = int(np.sum(binary_mask))
                
                # Add annotation
                coco_dict['annotations'].append({
                    'id': annotation_id,
                    'image_id': img_id,
                    'category_id': 1,  # Assuming one class for simplicity
                    'segmentation': segmentation,
                    'area': area,
                    'bbox': [x, y, w, h],
                    'iscrowd': 0
                })
                
                annotation_id += 1


def create_config_for_finetuning(base_config_path, custom_config_path, dataset_dir, num_classes=1):
    """Create a custom config file for fine-tuning"""
    # Load the base config
    cfg = Config.fromfile(base_config_path)
    
    # Modify for custom dataset
    cfg.dataset_type = 'CocoDataset'
    cfg.data_root = dataset_dir
    
    # Update dataset paths
    cfg.data.train.type = 'CocoDataset'
    cfg.data.train.ann_file = os.path.join(dataset_dir, 'train.json')
    cfg.data.train.img_prefix = os.path.join(dataset_dir, 'train')
    
    cfg.data.val.type = 'CocoDataset'
    cfg.data.val.ann_file = os.path.join(dataset_dir, 'val.json')
    cfg.data.val.img_prefix = os.path.join(dataset_dir, 'val')
    
    cfg.data.test.type = 'CocoDataset'
    cfg.data.test.ann_file = os.path.join(dataset_dir, 'val.json')
    cfg.data.test.img_prefix = os.path.join(dataset_dir, 'val')
    
    # Update number of classes (add background)
    cfg.model.roi_head.bbox_head.num_classes = num_classes
    cfg.model.roi_head.mask_head.num_classes = num_classes
    
    # Update training parameters
    cfg.optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
    cfg.optimizer_config = dict(grad_clip=None)
    cfg.lr_config = dict(
        policy='step',
        warmup='linear',
        warmup_iters=500,
        warmup_ratio=0.001,
        step=[8, 11]
    )
    cfg.runner = dict(type='EpochBasedRunner', max_epochs=12)
    
    # Update log config for TensorBoard
    cfg.log_config = dict(
        interval=50,
        hooks=[
            dict(type='TextLoggerHook'),
            dict(type='TensorboardLoggerHook')
        ]
    )
    
    # Save the modified config
    with open(custom_config_path, 'w') as f:
        f.write(cfg.pretty_text)
    
    return cfg


def train_model(config_path, work_dir, epochs=12):
    """Train the model with the specified config"""
    # Load config
    cfg = Config.fromfile(config_path)
    
    # Update work directory
    cfg.work_dir = work_dir
    os.makedirs(work_dir, exist_ok=True)
    
    # Set max epochs
    cfg.runner.max_epochs = epochs
    
    # Build dataset
    datasets = [build_dataset(cfg.data.train)]
    
    # Build model
    model = build_detector(cfg.model)
    model.init_weights()
    
    # Create work_dir
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    
    # Train model
    train_detector(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=True,
        timestamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    )


def evaluate_model(config_path, checkpoint_path, output_dir):
    """Evaluate the trained model on the validation set"""
    # Load config
    cfg = Config.fromfile(config_path)
    
    # Build the dataset
    dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    
    # Build the model
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    
    # Convert model to evaluation mode
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    
    # Evaluate and show the results
    outputs = mmcv.track_progress(inference_detector, [model, dataset])
    
    # Calculate and print mAP
    eval_results = dataset.evaluate(outputs, metric='bbox')
    print("Evaluation results:")
    for metric_name, value in eval_results.items():
        print(f"{metric_name}: {value}")


def main():
    args = parse_args()
    
    # Import mmdet modules (do this here to avoid slow import time if not needed)
    if args.task == 'inference':
        # Initialize the detector
        if args.checkpoint is None:
            print("Error: Checkpoint is required for inference")
            return
        
        model = init_detector(args.config, args.checkpoint, device=args.device)
        perform_inference(model, args.input, args.output, args.score_threshold)
    
    elif args.task == 'convert':
        if args.custom_dataset is None or args.custom_dataset_output is None:
            print("Error: --custom_dataset and --custom_dataset_output are required for conversion")
            return
        
        convert_to_coco_format(args.custom_dataset, args.custom_dataset_output)
    
    elif args.task == 'train':
        if args.custom_dataset_output is None:
            print("Error: --custom_dataset_output is required for training")
            return
        
        # Create custom config
        custom_config_path = os.path.join(args.output, 'custom_mask_rcnn_config.py')
        create_config_for_finetuning(args.config, custom_config_path, args.custom_dataset_output)
        
        # Train model
        train_model(custom_config_path, args.output, args.epochs)
    
    elif args.task == 'evaluate':
        if args.checkpoint is None or args.custom_dataset_output is None:
            print("Error: --checkpoint and --custom_dataset_output are required for evaluation")
            return
        
        # Create custom config if it doesn't exist
        custom_config_path = os.path.join(args.output, 'custom_mask_rcnn_config.py')
        if not os.path.exists(custom_config_path):
            create_config_for_finetuning(args.config, custom_config_path, args.custom_dataset_output)
        
        # Evaluate model
        evaluate_model(custom_config_path, args.checkpoint, args.output)


if __name__ == '__main__':
    main() 