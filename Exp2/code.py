import numpy as np

# Activation Function and its Derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class MLP_XOR:
    def __init__(self, input_size, hidden_size, output_size, lr=0.1, epochs=10000):
        # Initialize weights and biases
        self.lr = lr
        self.epochs = epochs
        self.hidden_weights = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.hidden_bias = np.random.uniform(-1, 1, (1, hidden_size))
        self.output_weights = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.output_bias = np.random.uniform(-1, 1, (1, output_size))

    def train(self, X, y):
        for epoch in range(self.epochs):
            # Forward Pass
            hidden_input = np.dot(X, self.hidden_weights) + self.hidden_bias
            hidden_output = sigmoid(hidden_input)
            final_input = np.dot(hidden_output, self.output_weights) + self.output_bias
            final_output = sigmoid(final_input)

            # Compute Error
            error = y - final_output

            # Backpropagation
            output_delta = error * sigmoid_derivative(final_output)
            hidden_delta = output_delta.dot(self.output_weights.T) * sigmoid_derivative(hidden_output)

            # Update Weights and Biases
            self.output_weights += hidden_output.T.dot(output_delta) * self.lr
            self.output_bias += np.sum(output_delta, axis=0, keepdims=True) * self.lr
            self.hidden_weights += X.T.dot(hidden_delta) * self.lr
            self.hidden_bias += np.sum(hidden_delta, axis=0, keepdims=True) * self.lr

    def predict(self, X):
        hidden_input = np.dot(X, self.hidden_weights) + self.hidden_bias
        hidden_output = sigmoid(hidden_input)
        final_input = np.dot(hidden_output, self.output_weights) + self.output_bias
        final_output = sigmoid(final_input)
        return np.round(final_output)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy

# XOR Dataset
X_XOR = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_XOR = np.array([[0], [1], [1], [0]])

# Train MLP
mlp_xor = MLP_XOR(input_size=2, hidden_size=2, output_size=1)
mlp_xor.train(X_XOR, y_XOR)
accuracy_xor = mlp_xor.evaluate(X_XOR, y_XOR)
print(f"XOR MLP Accuracy: {accuracy_xor * 100:.2f}%")

# Predictions
print("Predictions:")
for x, y in zip(X_XOR, mlp_xor.predict(X_XOR)):
    print(f"Input: {x} -> Predicted: {y[0]}")
