# LeNet CNN and Architecture Variants for MNIST - Experiment Report

## Introduction

This report presents the results of implementing the LeNet CNN architecture on the MNIST dataset and analyzing the impact of various architectural modifications on model performance.

## Experimental Setup

- **Dataset**: MNIST handwritten digits (60,000 training and 10,000 test samples)
- **Base Architecture**: LeNet-5 CNN
- **Training Details**:
  - Batch size: 64
  - Optimizer: Adam with learning rate 0.001
  - Loss function: Cross Entropy Loss
  - Number of epochs: 5
  - Hardware: [Fill in your hardware details]

## Experiments and Results

### 1. Effect of Activation Functions

| Model | Activation | Final Test Accuracy | Training Time (avg) |
|-------|------------|---------------------|---------------------|
| LeNet | ReLU       | [Fill in]           | [Fill in]           |
| LeNet | Sigmoid    | [Fill in]           | [Fill in]           |
| LeNet | Tanh       | [Fill in]           | [Fill in]           |
| LeNet | LeakyReLU  | [Fill in]           | [Fill in]           |

**Observations**:
- [Note which activation function performed best]
- [Note training speed differences]
- [Note any interesting convergence behavior]

### 2. Effect of Pooling Methods

| Model | Pooling Method | Final Test Accuracy | Training Time (avg) |
|-------|---------------|---------------------|---------------------|
| LeNet | Max Pooling   | [Fill in]           | [Fill in]           |
| LeNet | Avg Pooling   | [Fill in]           | [Fill in]           |
| LeNet | No Pooling    | [Fill in]           | [Fill in]           |

**Observations**:
- [Note differences between pooling methods]
- [Discuss implications of removing pooling]
- [Note computational efficiency differences]

### 3. Effect of Fully Connected Layers

| Model                | Final Test Accuracy | Training Time (avg) |
|----------------------|---------------------|---------------------|
| LeNet (standard)     | [Fill in]           | [Fill in]           |
| LeNet (reduced FC)   | [Fill in]           | [Fill in]           |

**Observations**:
- [Note the impact of reducing FC layers]
- [Discuss parameter count differences]
- [Note any trade-offs observed]

## Learning Curves

[Insert learning curve plots here]

## Best and Worst Configurations

### Top 5 Performing Configurations:
1. [Fill in model name and configuration]: [Accuracy]
2. [Fill in model name and configuration]: [Accuracy]
3. [Fill in model name and configuration]: [Accuracy]
4. [Fill in model name and configuration]: [Accuracy]
5. [Fill in model name and configuration]: [Accuracy]

### Worst 3 Performing Configurations:
1. [Fill in model name and configuration]: [Accuracy]
2. [Fill in model name and configuration]: [Accuracy]
3. [Fill in model name and configuration]: [Accuracy]

## Discussion

### Role of Activation Functions
[Elaborate on your findings regarding activation functions and their impact]

### Importance of Pooling Layers
[Elaborate on your findings regarding pooling layers and their impact]

### Value of Fully Connected Layers
[Elaborate on your findings regarding FC layers and their impact]

### Overall Architecture Insights
[General insights from this study]

## Conclusion

[Summarize the key findings and insights about what makes a CNN architecture effective]

## References

1. LeCun, Y., et al. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.
2. [Add other references as needed] 