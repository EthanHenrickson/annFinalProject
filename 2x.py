import numpy as np
import random as rd

import matplotlib.pyplot as plt


class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = 0.01 * np.random.randn(1, n_neurons)

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Loss_MeanSquaredError:
    # Returns overall data loss of model
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred) ** 2, axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples


def normalize(values):
    min_value = np.min(values)
    max_value = np.max(values)

    normalized_values = (values - min_value) / (max_value - min_value)

    return normalized_values


x = []
for i in range(0, 500):
    x.append([10 / 500 * i - 5])
x = np.array(x)
y = 2 * x
x = normalize(x)


layer1 = Layer_Dense(1, 10)
activation1 = Activation_ReLU()

layer2 = Layer_Dense(10, 1)
activation2 = Activation_ReLU()

layer3 = Layer_Dense(1, 1)

loss_function = Loss_MeanSquaredError()

epochs = 40000
learning_rate = 0.01
for epoch in range(epochs):
    # Forward pass
    layer1.forward(x)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)
    layer3.forward(activation2.output)
    # Calculate loss
    loss = loss_function.calculate(layer3.output, y)

    if loss < 0.01:
        learning_rate = 0.0025

    loss_function.backward(layer3.output, y)
    layer3.backward(loss_function.dinputs)
    activation2.backward(layer3.dinputs)
    layer2.backward(activation2.dinputs)
    activation1.backward(layer2.dinputs)
    layer1.backward(activation1.dinputs)
    # Update weights and biases
    layer1.weights -= learning_rate * layer1.dweights
    layer1.biases -= learning_rate * layer1.dbiases
    layer2.weights -= learning_rate * layer2.dweights
    layer2.biases -= learning_rate * layer2.dbiases
    layer3.weights -= learning_rate * layer3.dweights
    layer3.biases -= learning_rate * layer3.dbiases
    if epoch % 100 == 0:
        print(f"{loss:.4f}")
# Test the trained model


test_input = np.linspace(-5, 5, 500).reshape(-1, 1)

layer1.forward(normalize(test_input))
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)
layer3.forward(activation2.output)

loss = loss_function.calculate(layer3.output, test_input * 2)
prediction = layer3.output

plt.plot(test_input, 2 * test_input)
plt.plot(test_input, prediction)


plt.show()

print(loss)
