#!/usr/bin/env python3
"""
YOLOv4 Object Detection with Darknet

This script implements object detection using YOLOv4 with Darknet framework.
It provides functionality for:
1. Setting up Darknet and YOLOv4
2. Detecting objects in images
3. Detecting objects in videos
4. Optional training on custom datasets
"""

import os
import sys
import cv2
import numpy as np
import random
import subprocess
import argparse
from PIL import Image
import time
import glob

# Check if script is running in Google Colab
try:
    import google.colab
    IN_COLAB = True
    from google.colab.patches import cv2_imshow
    from google.colab import files
except ImportError:
    IN_COLAB = False
    def cv2_imshow(img):
        """Local implementation of cv2_imshow for non-Colab environments"""
        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def run_command(command):
    """Run shell command and print output"""
    print(f"Running: {command}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, universal_newlines=True)
    
    # Print output in real-time
    for line in iter(process.stdout.readline, ""):
        print(line.strip())
    
    for line in iter(process.stderr.readline, ""):
        print(line.strip())
    
    process.stdout.close()
    process.stderr.close()
    return_code = process.wait()
    
    if return_code != 0:
        print(f"Command failed with return code {return_code}")
    
    return return_code == 0

def setup_darknet():
    """Set up Darknet and YOLOv4"""
    print("Setting up Darknet and YOLOv4...")
    
    # Check if CUDA is available
    if not IN_COLAB:
        try:
            gpu_info = subprocess.check_output(['nvidia-smi']).decode('utf-8')
            print("GPU is available:")
            print(gpu_info)
        except Exception:
            print("GPU is not available. The detection will be slow on CPU.")
    
    # Clone Darknet repository if it doesn't exist
    if not os.path.exists('darknet'):
        success = run_command("git clone https://github.com/AlexeyAB/darknet")
        if not success:
            print("Failed to clone darknet repository")
            return False
    
    # Navigate to darknet directory
    os.chdir('darknet')
    
    # Update Makefile to use GPU and OpenCV
    if IN_COLAB or os.path.exists('/usr/local/cuda'):
        run_command("sed -i 's/OPENCV=0/OPENCV=1/' Makefile")
        run_command("sed -i 's/GPU=0/GPU=1/' Makefile")
        run_command("sed -i 's/CUDNN=0/CUDNN=1/' Makefile")
        run_command("sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile")
    else:
        # If no GPU, just enable OpenCV
        run_command("sed -i 's/OPENCV=0/OPENCV=1/' Makefile")
    
    # Compile darknet
    success = run_command("make")
    if not success:
        print("Failed to compile darknet")
        return False
    
    # Download YOLOv4 pre-trained weights if they don't exist
    if not os.path.exists('yolov4.weights'):
        print("Downloading YOLOv4 weights...")
        success = run_command("wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights")
        if not success:
            print("Failed to download YOLOv4 weights")
            return False
    
    print("Darknet and YOLOv4 setup completed successfully")
    return True

def detect_image(image_path, threshold=0.3, show_image=True):
    """Detect objects in an image"""
    print(f"Detecting objects in image: {image_path}")
    
    # Run darknet detection
    cmd = f"./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights {image_path} -thresh {threshold}"
    run_command(cmd)
    
    # Display results
    if show_image and os.path.exists('predictions.jpg'):
        image = cv2.imread('predictions.jpg')
        if image is not None:
            print("Displaying detection results")
            cv2_imshow(image)
        else:
            print("Failed to read prediction image")
    
    return 'predictions.jpg' if os.path.exists('predictions.jpg') else None

def detect_custom_image(threshold=0.3):
    """Upload and detect objects in a custom image"""
    if not IN_COLAB:
        print("This function is only available in Google Colab")
        return None
    
    print("Please upload an image")
    uploaded = files.upload()
    
    for filename in uploaded.keys():
        print(f"Uploaded file: {filename}")
        
        # Display original image
        img = Image.open(filename)
        img.thumbnail((600, 600))
        display(img)
        
        # Run detection
        return detect_image(filename, threshold)
    
    return None

def detect_video(video_path, threshold=0.3, output_filename="result.avi"):
    """Detect objects in a video"""
    print(f"Detecting objects in video: {video_path}")
    
    # Run darknet detection on video
    cmd = f"./darknet detector demo cfg/coco.data cfg/yolov4.cfg yolov4.weights {video_path} -out_filename {output_filename} -thresh {threshold} -dont_show"
    run_command(cmd)
    
    # Convert to MP4 format
    mp4_filename = output_filename.replace('.avi', '.mp4')
    if os.path.exists(output_filename):
        run_command(f"ffmpeg -i {output_filename} -vcodec libx264 {mp4_filename}")
        print(f"Converted video saved as {mp4_filename}")
        return mp4_filename
    else:
        print("Video processing failed")
        return None

def detect_custom_video(threshold=0.3):
    """Upload and detect objects in a custom video"""
    if not IN_COLAB:
        print("This function is only available in Google Colab")
        return None
    
    print("Please upload a video")
    uploaded = files.upload()
    
    for filename in uploaded.keys():
        print(f"Uploaded file: {filename}")
        return detect_video(filename, threshold, "custom_result.avi")
    
    return None

def prepare_custom_dataset():
    """Prepare directory structure for custom dataset"""
    print("Preparing custom dataset directories...")
    
    os.makedirs('custom_dataset/images', exist_ok=True)
    os.makedirs('custom_dataset/labels', exist_ok=True)
    
    if IN_COLAB:
        print("Please upload training images (jpg, png)")
        uploaded_images = files.upload()
        
        for fn in uploaded_images.keys():
            with open(f'custom_dataset/images/{fn}', 'wb') as f:
                f.write(uploaded_images[fn])
        
        print("Please upload annotation labels (txt files)")
        uploaded_labels = files.upload()
        
        for fn in uploaded_labels.keys():
            with open(f'custom_dataset/labels/{fn}', 'wb') as f:
                f.write(uploaded_labels[fn])
    else:
        print("Please manually copy your images to custom_dataset/images/")
        print("Please manually copy your label files to custom_dataset/labels/")
        input("Press Enter to continue when done...")
    
    # Create classes.names file
    print("Creating classes.names file")
    class_names = []
    while True:
        class_name = input("Enter class name (or press enter to finish): ")
        if not class_name:
            break
        class_names.append(class_name)
    
    if not class_names:
        print("No classes specified. Adding placeholder classes.")
        class_names = ["class1", "class2"]
    
    with open('custom_dataset/classes.names', 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    
    print(f"Created classes.names with {len(class_names)} classes")
    return len(class_names)

def prepare_training_files(num_classes):
    """Prepare configuration files for training"""
    print("Preparing training configuration files...")
    
    # Create custom.data file
    with open('custom_dataset/custom.data', 'w') as f:
        f.write(f'classes = {num_classes}\n')
        f.write('train = custom_dataset/train.txt\n')
        f.write('valid = custom_dataset/valid.txt\n')
        f.write('names = custom_dataset/classes.names\n')
        f.write('backup = backup/\n')
    
    # Create list of training images
    image_paths = glob.glob('custom_dataset/images/*.jpg')
    image_paths += glob.glob('custom_dataset/images/*.png')
    
    if not image_paths:
        print("No images found in custom_dataset/images/")
        return False
    
    # Split into train and validation sets
    random.shuffle(image_paths)
    split_point = int(len(image_paths) * 0.9)
    train_paths = image_paths[:split_point]
    valid_paths = image_paths[split_point:]
    
    with open('custom_dataset/train.txt', 'w') as f:
        for path in train_paths:
            f.write(f"{path}\n")
    
    with open('custom_dataset/valid.txt', 'w') as f:
        for path in valid_paths:
            f.write(f"{path}\n")
    
    print(f"Created train.txt with {len(train_paths)} images")
    print(f"Created valid.txt with {len(valid_paths)} images")
    
    # Copy and modify YOLOv4 config file
    run_command("cp cfg/yolov4-custom.cfg custom_dataset/yolov4-custom.cfg")
    
    # Update configuration parameters
    with open('custom_dataset/yolov4-custom.cfg', 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if 'batch=' in line:
            lines[i] = 'batch=64\n'
        elif 'subdivisions=' in line:
            lines[i] = 'subdivisions=16\n'
        elif 'max_batches =' in line:
            lines[i] = f'max_batches = {max(6000, num_classes*2000)}\n'
        elif 'steps=' in line:
            max_batches = max(6000, num_classes*2000)
            steps1 = int(max_batches * 0.8)
            steps2 = int(max_batches * 0.9)
            lines[i] = f'steps={steps1},{steps2}\n'
        elif 'classes=' in line:
            lines[i] = f'classes={num_classes}\n'
        elif 'filters=' in line and '[yolo]' in lines[i-1:i-2:-1]:
            lines[i] = f'filters={(num_classes + 5) * 3}\n'
    
    with open('custom_dataset/yolov4-custom.cfg', 'w') as f:
        f.writelines(lines)
    
    print("Modified yolov4-custom.cfg for custom training")
    return True

def train_custom_model():
    """Train custom YOLOv4 model"""
    print("Starting custom model training...")
    
    # Create backup directory
    os.makedirs('backup', exist_ok=True)
    
    # Download pre-trained weights for transfer learning
    if not os.path.exists('yolov4.conv.137'):
        print("Downloading pre-trained weights for transfer learning...")
        run_command("wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137")
    
    # Start training
    print("Training may take a long time. You can stop it with Ctrl+C and resume later.")
    run_command("./darknet detector train custom_dataset/custom.data custom_dataset/yolov4-custom.cfg yolov4.conv.137 -dont_show")
    
    return True

def test_custom_model(image_path, threshold=0.3):
    """Test custom trained model on an image"""
    print(f"Testing custom model on image: {image_path}")
    
    if not os.path.exists('backup/yolov4-custom_best.weights'):
        print("Custom model weights not found in backup/yolov4-custom_best.weights")
        return None
    
    # Run detection with custom model
    cmd = f"./darknet detector test custom_dataset/custom.data custom_dataset/yolov4-custom.cfg backup/yolov4-custom_best.weights {image_path} -thresh {threshold}"
    run_command(cmd)
    
    # Display results
    if os.path.exists('predictions.jpg'):
        image = cv2.imread('predictions.jpg')
        if image is not None:
            print("Displaying detection results")
            cv2_imshow(image)
        else:
            print("Failed to read prediction image")
    
    return 'predictions.jpg' if os.path.exists('predictions.jpg') else None

def evaluate_model():
    """Calculate mAP for custom model"""
    print("Evaluating custom model...")
    
    if not os.path.exists('backup/yolov4-custom_best.weights'):
        print("Custom model weights not found in backup/yolov4-custom_best.weights")
        return False
    
    # Calculate mAP
    run_command("./darknet detector map custom_dataset/custom.data custom_dataset/yolov4-custom.cfg backup/yolov4-custom_best.weights")
    
    return True

def main():
    """Main function to parse arguments and run functions"""
    parser = argparse.ArgumentParser(description='YOLOv4 Object Detection with Darknet')
    
    parser.add_argument('--setup', action='store_true', help='Setup Darknet and YOLOv4')
    parser.add_argument('--detect-image', type=str, help='Path to image for detection')
    parser.add_argument('--detect-video', type=str, help='Path to video for detection')
    parser.add_argument('--threshold', type=float, default=0.3, help='Detection threshold (default: 0.3)')
    parser.add_argument('--custom-train', action='store_true', help='Train custom YOLOv4 model')
    parser.add_argument('--custom-test', type=str, help='Test custom model on image')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate custom model (mAP)')
    
    if len(sys.argv) == 1:
        # If no arguments, run in interactive mode
        parser.print_help()
        interactive_mode()
        return
    
    args = parser.parse_args()
    
    # Change directory to script location
    script_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_dir)
    
    if args.setup:
        setup_darknet()
    
    if args.detect_image:
        detect_image(args.detect_image, args.threshold)
    
    if args.detect_video:
        detect_video(args.detect_video, args.threshold)
    
    if args.custom_train:
        os.chdir('darknet')
        num_classes = prepare_custom_dataset()
        prepare_training_files(num_classes)
        train_custom_model()
    
    if args.custom_test:
        os.chdir('darknet')
        test_custom_model(args.custom_test, args.threshold)
    
    if args.evaluate:
        os.chdir('darknet')
        evaluate_model()

def interactive_mode():
    """Run in interactive menu mode"""
    current_dir = os.getcwd()
    
    while True:
        print("\nYOLOv4 Object Detection with Darknet - Interactive Mode")
        print("1. Setup Darknet and YOLOv4")
        print("2. Detect objects in an image")
        print("3. Detect objects in a video")
        print("4. Train custom YOLOv4 model")
        print("5. Test custom model")
        print("6. Evaluate custom model (mAP)")
        print("0. Exit")
        
        choice = input("Enter your choice (0-6): ")
        
        if choice == '0':
            break
        
        elif choice == '1':
            setup_darknet()
            os.chdir(current_dir)
        
        elif choice == '2':
            os.chdir(current_dir)
            if os.path.exists('darknet'):
                os.chdir('darknet')
                if IN_COLAB:
                    detect_custom_image()
                else:
                    image_path = input("Enter path to image: ")
                    detect_image(image_path)
            else:
                print("Darknet is not set up. Please run setup first.")
            os.chdir(current_dir)
        
        elif choice == '3':
            os.chdir(current_dir)
            if os.path.exists('darknet'):
                os.chdir('darknet')
                if IN_COLAB:
                    detect_custom_video()
                else:
                    video_path = input("Enter path to video: ")
                    detect_video(video_path)
            else:
                print("Darknet is not set up. Please run setup first.")
            os.chdir(current_dir)
        
        elif choice == '4':
            os.chdir(current_dir)
            if os.path.exists('darknet'):
                os.chdir('darknet')
                num_classes = prepare_custom_dataset()
                prepare_training_files(num_classes)
                train_custom_model()
            else:
                print("Darknet is not set up. Please run setup first.")
            os.chdir(current_dir)
        
        elif choice == '5':
            os.chdir(current_dir)
            if os.path.exists('darknet'):
                os.chdir('darknet')
                if not os.path.exists('backup/yolov4-custom_best.weights'):
                    print("Custom model weights not found. Please train the model first.")
                else:
                    image_path = input("Enter path to test image: ")
                    test_custom_model(image_path)
            else:
                print("Darknet is not set up. Please run setup first.")
            os.chdir(current_dir)
        
        elif choice == '6':
            os.chdir(current_dir)
            if os.path.exists('darknet'):
                os.chdir('darknet')
                evaluate_model()
            else:
                print("Darknet is not set up. Please run setup first.")
            os.chdir(current_dir)
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 