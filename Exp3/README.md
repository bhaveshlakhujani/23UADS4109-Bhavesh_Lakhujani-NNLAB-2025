# Experiment 3

# Description of code
It is a three-Layer Fully Connected Neural Network (MLP) for classifying MNIST handwritten digits (0-9).

## Model Architecture
Input Layer (784 neurons):
Each 28x28 grayscale image is flattened into a 1D vector of 784 pixels.

Hidden Layer 1 (256 neurons, ReLU Activation):
Fully connected layer with 256 neurons.
Uses ReLU activation to for non-linearity.

Hidden Layer 2 (256 neurons, ReLU Activation): 
Next fully connected layer with 256 neurons and ReLU activation.

Output Layer (10 neurons): 
Fully connected layer with 10 neurons.

### Training Process

Loss Function: Softmax Cross-Entropy measures the difference between predicted and actual labels.

Optimizer: Adam Optimizer adjusts weights and biases using backpropagation to minimize the loss.


# My Comments
Very high number of input features directly, so number of parameters is high.
It is complex network (fully connected MLP), may be it ovrfits the training data.

We can either use CNN for Image Classification.
We can use early stoping to prevent from overfitting.
