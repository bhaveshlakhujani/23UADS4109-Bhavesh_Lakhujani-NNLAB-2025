# Objective: WAP to implement a multi-layer perceptron (MLP) network with one hidden layer using numpy in Python. 
# Demonstrate that it can learn the XOR Boolean function.


'''Description of the Model
This implementation is a Multi-Layer Perceptron (MLP) neural network with:

An input layer (2 neurons) → Represents the two input values (X1, X2) of the XOR function.
One hidden layer (4 neurons) → Introduces non-linearity to handle XOR (which is not linearly separable).
An output layer (1 neuron) → Outputs the final prediction (either 0 or 1).
The step function is used as the activation function. It maps the neuron output to either 0 or 1, ensuring binary classification.
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, epochs=10000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # Initialize weights and biases
        np.random.seed(42)
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))
    
    def step_function(self, x):
        return np.where(x >= 0, 1, 0)
    
    def train(self, X, y):
        for epoch in range(self.epochs):
            hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
            hidden_layer_output = self.step_function(hidden_layer_input)
            output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_output
            output_layer_output = self.step_function(output_layer_input)
            
            error = y - output_layer_output
            
            # Weight and bias updates
            self.weights_input_hidden += np.dot(X.T, np.dot(error, self.weights_hidden_output.T)) * self.learning_rate
            self.weights_hidden_output += np.dot(hidden_layer_output.T, error) * self.learning_rate
            self.bias_output += np.sum(error, axis=0, keepdims=True) * self.learning_rate
            self.bias_hidden += np.sum(np.dot(error, self.weights_hidden_output.T), axis=0, keepdims=True) * self.learning_rate
            
    def predict(self, X):
        hidden_layer_activation = self.step_function(np.dot(X, self.weights_input_hidden) + self.bias_hidden)
        final_output = self.step_function(np.dot(hidden_layer_activation, self.weights_hidden_output) + self.bias_output)
        return final_output
    
    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        print(f"\nFinal Accuracy: {accuracy * 100:.2f}%")
        return predictions

# XOR 
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])


mlp = MLP(input_size=2, hidden_size=4, output_size=1, learning_rate=0.01, epochs=100)
mlp.train(X, y)


final_output = mlp.evaluate(X, y)


print("\nFinal Predictions for XOR:")
for i in range(len(X)):
    print(f"Input: {X[i]} → Predicted Output: {final_output[i][0]}")


'''
Description of Code:
1) Define a  class of MLP
2) Initialize input, hidden, output layer sizes.
3) Initializes weights randomly and bias to be zero.
4)using the step activation function.
5) define a train function in MLP class in which:
    Performs forward propagation to compute hidden and output layers.
    Computes error
    Manually updates weights and biases without using backpropagation.
    Train the model and return the predictions with its accuracy.
'''

'''
My Comments
Limitations:
1) In this experiment, we are updating the weights and biases manually without using the backpropagation, that's why it is specific to a particular problem. 
2) Not using any optimization technique
Scope of Improvment:
1) for updating the weights and bias automatically we can use gradient descent in which we define a formula to update the weights and bias instead of
mannually choosing it.
2)Use of nonlinear activation function like sigmoid, ReLU
'''
