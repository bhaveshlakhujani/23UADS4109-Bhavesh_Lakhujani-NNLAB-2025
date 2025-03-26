## Description of the Model   
The model is a Convolutional Neural Network (CNN) designed for classifying the Fashion MNIST dataset. It consists of:    
1) Convolutional Layers (Extract spatial features from images)     
2) Max-Pooling Layers (Reduce spatial dimensions)     
3) Dropout Layers (Prevent overfitting)     
4) Fully Connected Layers (For classification)     
5) Softmax Output Layer (Produces probability distribution for 10 classes)
   
The model is trained using different hyperparameters (filter size, batch size, optimizer, and regularization) to analyze their impact on performance.


##  Description of the Code     
Data Loading & Preprocessing      

1) Loads Fashion MNIST dataset.    

2) Normalizes images (divides by 255).     

3) Reshapes images to (28, 28, 1) to match CNN input requirements.     

Model Creation Function     

1) Uses Conv2D layers with varying filter sizes.

2) Applies MaxPooling2D to downsample features.     

3) Uses Dropout (0.1) and optional L2 regularization (0.001) to reduce overfitting.   

4) Flattens the features and passes them through fully connected layers.   

5) Outputs a softmax layer with 10 units (for 10 fashion categories).

Hyperparameter Experimentation

1) Tests different filter sizes (3, 5).

2) Compares batch sizes (128, 256).

3) Uses Adam and SGD optimizers.

4) Evaluates the impact of L2 regularization.

5) Trains models for 30 epochs without verbose output.

Result Logging

1) Suppresses training details.

2) Prints a summary per model, showing hyperparameters and final accuracy.

