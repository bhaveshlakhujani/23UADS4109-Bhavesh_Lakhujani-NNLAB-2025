# Objective: WAP to implement a multi-layer perceptron (MLP) network with one hidden layer using numpy in Python. 
# Demonstrate that it can learn the XOR Boolean function.

import numpy as np

def step_function(x):
    return 1 if x >= 0 else 0

def forward_pass(X, W1, B1, W2, B2):
    
    hidden_input = np.dot(X, W1) + B1
    hidden_output = np.array([step_function(x) for x in hidden_input])
    final_input = np.dot(hidden_output, W2) + B2
    final_output = step_function(final_input)

    return final_output
    
X_XOR = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_XOR = np.array([0, 1, 1, 0]) 

W1 = np.array([
    [1,  1, -1, -1],  
    [1, -1,  1, -1]   
])

B1 = np.array([-0.5, -0.5, -0.5, -0.5])  

W2 = np.array([1, 1, 1, 1])  
B2 = -2  #


predictions = [forward_pass(x, W1, B1, W2, B2) for x in X_XOR]


correct_predictions = sum(p == y for p, y in zip(predictions, y_XOR))
accuracy = (correct_predictions / len(y_XOR)) * 100

print("\nMLP Output for XOR Function (With Manual Weight Updates):")
for i, (x, y) in enumerate(zip(X_XOR, predictions)):
    print(f"Input: {x} → Predicted Output: {y}, Actual Output: {y_XOR[i]}")

print(f"\nModel Accuracy: {accuracy:.2f}%")

'''
My Comments
Limitations:
1) In this program, we are choosing the weights and biases manually, that's why it is specific to a particular problem. If we change the values 
of input features (like from XOR to NOR), it will not work good.

Scope of Improvment:
1) for updating the weights and bias automatically we can use gradient descent in which we define a formula to update the weights and bias instead of
mannually choosing it.
2) For learning and training the neural network we can use ReLU function, tanh, sigmoid function instead of step function.
'''
