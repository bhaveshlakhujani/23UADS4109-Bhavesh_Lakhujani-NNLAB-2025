import numpy as np

class Perceptron:
    def __init__(self, input_size, lr=0.1, epochs=100):
        self.weights = np.zeros(input_size + 1)  # +1 for bias term
        self.lr = lr
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        x = np.insert(x, 0, 1)  # Adding bias term
        return self.activation(np.dot(self.weights, x))

    def train(self, X, y):
        for epoch in range(self.epochs):
            for i in range(len(X)):
                x_i = np.insert(X[i], 0, 1)  # Adding bias term
                y_pred = self.activation(np.dot(self.weights, x_i))
                error = y[i] - y_pred
                self.weights += self.lr * error * x_i  # Weight update rule

    def evaluate(self, X, y):
        correct = sum(self.predict(x) == y_i for x, y_i in zip(X, y))
        accuracy = correct / len(y)
        return accuracy


X_NAND = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_NAND = np.array([1, 1, 1, 0])


X_XOR = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_XOR = np.array([0, 1, 1, 0])


perceptron_nand = Perceptron(input_size=2)
perceptron_nand.train(X_NAND, y_NAND)
accuracy_nand = perceptron_nand.evaluate(X_NAND, y_NAND)
print(f"NAND Perceptron Accuracy: {accuracy_nand * 100:.2f}%")


perceptron_xor = Perceptron(input_size=2)
perceptron_xor.train(X_XOR, y_XOR)
accuracy_xor = perceptron_xor.evaluate(X_XOR, y_XOR)
print(f"XOR Perceptron Accuracy: {accuracy_xor * 100:.2f}%")


print("NAND Perceptron Weights:", perceptron_nand.weights)
print("XOR Perceptron Weights:", perceptron_xor.weights)

'''
Explanation:
NAND Gate:
Linearly Separable: The perceptron can correctly classify inputs for a NAND gate because the problem is linearly separable. In simple terms, you can draw a straight line to separate the input values into their correct categories.

XOR Gate:
Non-Linearly Separable: The perceptron fails to classify an XOR gate correctly because the XOR function is not linearly separable. This means you can't draw a straight line to separate the input values into their correct categories.

Limitations:
Linearly Separable Problems: The perceptron algorithm works well for problems like NAND gates, which are linearly separable.

Non-Linearly Separable Problems: For problems like XOR gates, which are not linearly separable, a single perceptron is not enough. You need more complex networks, such as multi-layer neural networks (often referred to as multi-layer perceptrons or MLPs), to solve these non-linear problems.
'''
