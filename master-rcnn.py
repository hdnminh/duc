#!/usr/bin/env python
# coding: utf-8

"""
Mask R-CNN Model Setup and Demo
This script demonstrates:
- Setting up the MMDetection environment
- Loading a pre-trained Mask R-CNN model
- Running inference on a demo image 
- Visualizing the detection results
"""

import os
import sys
import subprocess
import torch
import mmcv
import mmengine
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.registry import VISUALIZERS
import matplotlib.pyplot as plt


def install_dependencies():
    """Install the required dependencies for MMDetection."""
    try:
        # Install dependencies
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "openmim"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "mmengine>=0.7.0"])
        
        # Install MMCV - CPU only version to avoid CUDA compilation issues
        subprocess.check_call([sys.executable, "-m", "pip", "install", "mmcv-full<2.2.0", "-f", "https://download.openmmlab.com/mmcv/dist/cpu/torch2.0.0/index.html"])
        
        # Clone mmdetection repository if not exists
        if not os.path.exists("mmdetection"):
            subprocess.check_call(["git", "clone", "https://github.com/open-mmlab/mmdetection.git"])
        
        # Change to mmdetection directory and install it
        current_dir = os.getcwd()
        os.chdir("mmdetection")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
        os.chdir(current_dir)
        
        # Download model checkpoint
        subprocess.check_call(["mim", "download", "mmdet", "--config", 
                             "mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco", 
                             "--dest", "./checkpoints"])
        
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)


def check_environment():
    """Check if all required packages are installed correctly."""
    try:
        # Check PyTorch installation
        print(f"PyTorch version: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
        
        # Check MMDetection installation
        import mmdet
        print(f"MMDetection version: {mmdet.__version__}")
        
        # Check MMCV installation
        import mmcv
        print(f"MMCV version: {mmcv.__version__}")
        
        # Check MMEngine installation
        import mmengine
        print(f"MMEngine version: {mmengine.__version__}")
        
        return True
    except ImportError as e:
        print(f"Error: {e}")
        return False


def setup_model(device='cpu'):
    """Initialize the Mask R-CNN model with pre-trained weights."""
    # Config and checkpoint files
    config_file = 'configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco.py'
    checkpoint_file = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
    
    # Register all modules in mmdet into the registries
    register_all_modules()
    
    # Build the model from config and checkpoint files
    try:
        model = init_detector(config_file, checkpoint_file, device=device)
        return model
    except Exception as e:
        print(f"Error setting up model: {e}")
        return None


def run_inference(model, image_path='demo/demo.jpg'):
    """Run inference on an image using the model."""
    try:
        # Load image
        image = mmcv.imread(image_path, channel_order='rgb')
        
        # Run inference
        result = inference_detector(model, image)
        print("Inference completed successfully")
        
        return image, result
    except Exception as e:
        print(f"Error during inference: {e}")
        return None, None


def visualize_result(model, image, result, output_file='output_detection.jpg'):
    """Visualize the detection results and save to file."""
    try:
        # Initialize the visualizer
        visualizer = VISUALIZERS.build(model.cfg.visualizer)
        visualizer.dataset_meta = model.dataset_meta
        
        # Show the results
        visualizer.add_datasample(
            'result',
            image,
            data_sample=result,
            draw_gt=None,
            wait_time=0,
        )
        
        # Save to file instead of showing
        os.makedirs('output', exist_ok=True)
        save_path = os.path.join('output', output_file)
        visualizer.get_image().save(save_path)
        print(f"Visualization saved to {save_path}")
    except Exception as e:
        print(f"Error during visualization: {e}")


def main():
    """Main function to run the demo."""
    print("Starting Mask R-CNN setup and demo...")
    
    # Fix for MMCV version incompatibility
    try:
        import mmcv
        from packaging import version
        mmcv_version = version.parse(mmcv.__version__)
        if mmcv_version >= version.parse("2.2.0"):
            print(f"Detected incompatible MMCV version: {mmcv.__version__}")
            print("Installing compatible MMCV version...")
            # Use CPU-only version to avoid CUDA compilation issues
            subprocess.check_call([sys.executable, "-m", "pip", "install", "mmcv-full<2.2.0", "-f", 
                                 "https://download.openmmlab.com/mmcv/dist/cpu/torch2.0.0/index.html"])
            print("MMCV downgraded successfully. Please restart the script.")
            sys.exit(0)
    except ImportError:
        pass
    
    # Install dependencies if needed
    if len(sys.argv) > 1 and sys.argv[1] == "--install":
        install_dependencies()
    
    # Check environment
    if not check_environment():
        print("Environment check failed. Please install the required dependencies.")
        sys.exit(1)
    
    # Setup model
    print("Setting up model...")
    # Always use CPU device to avoid CUDA issues
    device = 'cpu'
    model = setup_model(device)
    if model is None:
        print("Failed to set up model. Exiting.")
        sys.exit(1)
    
    # Run inference
    print("Running inference...")
    image, result = run_inference(model)
    if image is None or result is None:
        print("Inference failed. Exiting.")
        sys.exit(1)
    
    # Visualize results
    print("Visualizing results...")
    visualize_result(model, image, result)


if __name__ == "__main__":
    main()