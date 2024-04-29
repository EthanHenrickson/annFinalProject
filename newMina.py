import numpy as np

import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def linear(x):
    return x


class DenseLayer:
    def __init__(self, input_size, output_size, activation):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.activation = activation

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.activation(self.output)


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def test(network, X, y):
    # Forward pass
    output = network.forward(X)

    # Calculate loss
    loss = mean_squared_error(y, output)

    print(f"Test Loss: {loss:.4f}")

    return output


def train(network, X, y, epochs, learning_rate):
    for _ in range(epochs):
        # Forward pass
        output = network.forward(X)

        # Calculate loss
        loss = mean_squared_error(y, output)

        # Backward pass (backpropagation)
        # Calculate the gradient of the loss with respect to the output
        grad_output = 2 * (output - y) / y.size

        # Backpropagate the gradient through the layers
        for layer in reversed(network.layers):
            # Calculate the gradient of the loss with respect to the layer's inputs
            grad_input = np.dot(grad_output, layer.weights.T)

            # Calculate the gradient of the loss with respect to the layer's weights and biases
            grad_weights = np.dot(layer.inputs.T, grad_output)
            grad_biases = np.sum(grad_output, axis=0, keepdims=True)

            # Update the layer's weights and biases
            layer.weights -= learning_rate * grad_weights
            layer.biases -= learning_rate * grad_biases

            # Set the gradient for the next layer
            grad_output = grad_input * (
                layer.activation(layer.output) if layer.activation != linear else 1
            )

        # Print the loss for every 100 epochs
        if _ % 100 == 0:
            print(f"Epoch: {_}, Loss: {loss}")


# Create a neural network with two hidden layers
network = NeuralNetwork()
network.add_layer(DenseLayer(input_size=1, output_size=128, activation=linear))
network.add_layer(DenseLayer(input_size=128, output_size=128, activation=sigmoid))
network.add_layer(DenseLayer(input_size=128, output_size=1, activation=linear))

# Generate some sample data


def normalize(values):
    min_value = np.min(values)
    max_value = np.max(values)

    normalized_values = (values - min_value) / (max_value - min_value)

    return normalized_values


X_original = np.linspace(0, 15, 100).reshape(-1, 1)
X_original = np.random.rand(500,1) * 10 - 5

y = np.sin(X_original)
X = normalize(X_original)

# Train the neural network
train(network, X, y, epochs=50000, learning_rate=0.01)

print(X_original)
print(test(network,X, y))

plt.plot(X, y)
