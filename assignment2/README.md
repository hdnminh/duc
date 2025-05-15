# YOLOv4 Darknet Object Detection

This repository contains a Python script for object detection using YOLOv4 with the Darknet framework. The script provides both command-line and interactive interfaces for various object detection tasks.

## Features

- Set up Darknet and YOLOv4 with GPU support (CUDA)
- Detect objects in images using pre-trained YOLOv4 weights
- Detect objects in videos and generate output with bounding boxes
- Upload and process custom images and videos (in Google Colab)
- Train custom YOLOv4 models on your own datasets
- Evaluate model performance with mAP (mean Average Precision)

## Prerequisites

- Python 3.6 or higher
- OpenCV
- NumPy
- Git
- CUDA and cuDNN (for GPU acceleration, optional but recommended)
- FFmpeg (for video processing)

If running on Google Colab, all dependencies will be automatically installed.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/yolov4-detection.git
   cd yolov4-detection
   ```

2. Install required Python packages:
   ```
   pip install numpy opencv-python pillow
   ```

3. Make sure the script is executable:
   ```
   chmod +x yolov4_darknet_detection.py
   ```

## Usage

### Interactive Mode

Run the script without arguments to enter interactive mode:

```
./yolov4_darknet_detection.py
```

This will display a menu with options:
1. Setup Darknet and YOLOv4
2. Detect objects in an image
3. Detect objects in a video
4. Train custom YOLOv4 model
5. Test custom model
6. Evaluate custom model (mAP)

### Command-Line Mode

The script also supports command-line arguments for automation:

```
# Set up Darknet and YOLOv4
./yolov4_darknet_detection.py --setup

# Detect objects in an image
./yolov4_darknet_detection.py --detect-image path/to/image.jpg

# Detect objects in a video
./yolov4_darknet_detection.py --detect-video path/to/video.mp4

# Set detection threshold (default is 0.3)
./yolov4_darknet_detection.py --detect-image path/to/image.jpg --threshold 0.5

# Train a custom model
./yolov4_darknet_detection.py --custom-train

# Test a custom model
./yolov4_darknet_detection.py --custom-test path/to/test/image.jpg

# Evaluate a custom model
./yolov4_darknet_detection.py --evaluate
```

## Workflow for Custom Training

To train a custom YOLOv4 model:

1. Prepare your labeled dataset in YOLO format
2. Run `./yolov4_darknet_detection.py --custom-train`
3. Follow the prompts to:
   - Enter class names
   - Specify training images and label files
   - Configure training parameters
4. Wait for training to complete (can take several hours/days)
5. Test your custom model with `--custom-test`
6. Evaluate performance with `--evaluate`

## Google Colab Integration

The script automatically detects when running in Google Colab and enables additional features:
- File upload interface for images and videos
- Visualization tools for showing results
- Usage of Colab's GPU acceleration

## Troubleshooting

### Common Issues

1. **GPU not detected**
   - Make sure CUDA and cuDNN are properly installed
   - Check that your GPU is compatible with CUDA

2. **Build errors with Darknet**
   - Make sure you have build dependencies installed
   - On Ubuntu: `apt-get install build-essential`

3. **Video processing fails**
   - Install FFmpeg: `apt-get install ffmpeg`

4. **"darknet: command not found"**
   - The script should handle directories automatically, but if you encounter this error, make sure you're in the darknet directory when running detection commands

### Performance Tips

- Use GPU acceleration when possible for significantly faster detection
- For real-time detection, reduce image/video resolution
- Adjust threshold parameter to balance between precision and recall

## License

This project uses the Darknet framework which is open source under the [Darknet License](https://github.com/AlexeyAB/darknet/blob/master/LICENSE). 