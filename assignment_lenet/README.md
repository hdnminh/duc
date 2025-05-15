# LeNet CNN and Architecture Variants for MNIST

This project implements the LeNet CNN architecture for the MNIST dataset and performs experiments with various architecture modifications to analyze their impact on model performance.

## Prerequisites

- Python 3.6+
- PyTorch
- torchvision
- matplotlib
- numpy

Install dependencies:

```bash
pip install torch torchvision matplotlib numpy
```

## Project Structure

- `lenet_mnist.py`: Main Python script that implements the LeNet models, training, evaluation, and experiment methods

## Running the Experiments

To run all the experiments, simply execute:

```bash
python lenet_mnist.py
```

This will:
1. Download the MNIST dataset (if not already downloaded)
2. Train the original LeNet model with different activation functions and pooling layers
3. Train LeNet variants without pooling layers 
4. Train LeNet variants with reduced fully connected layers
5. Generate comparison plots and a summary of results

## Experiment Details

The code tests variations of the LeNet architecture with:

1. Different activation functions:
   - ReLU
   - Sigmoid
   - Tanh
   - LeakyReLU

2. Different pooling methods:
   - Max pooling
   - Average pooling

3. Architectural changes:
   - Standard LeNet architecture
   - LeNet without pooling layers (using strided convolutions instead)
   - LeNet with reduced fully connected layers

## Output

The script generates:

1. Training logs showing loss and accuracy for each epoch
2. Visualization plots:
   - `accuracy_comparison.png`: Test accuracy vs epochs for all model variants
   - `loss_comparison.png`: Test loss vs epochs for all model variants
   - `final_accuracy_comparison.png`: Bar chart of final test accuracies
   - `training_time_comparison.png`: Bar chart of training times
3. A summary of the best and worst performing model configurations

## Analysis

The results can be used to analyze:
- The impact of activation functions on model performance
- The importance of pooling layers in CNNs
- The role of fully connected layers in the classification process
- Performance vs computational efficiency trade-offs 