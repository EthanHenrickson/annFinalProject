import numpy as np
import random as rd
import math
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


x = np.linspace(-3, (5 * (math.pi)), 500).reshape(-1, 1)
y = np.sin(x)
print(x[1], y[1])

# A layer with 1 input and 16 output neurons
hiddenNodes1 = 32
layer1 = Layer_Dense(1, hiddenNodes1)
activation1 = Activation_ReLU()

# A second layer with 16 input and 8 output neurons

layer2 = Layer_Dense(hiddenNodes1, 1)
activation2 = Activation_ReLU()

# A third layer with 8 input and 1 output neurons

loss_function = Loss_MeanSquaredError()

epochs = 100000
learning_rate = 0.001

for epoch in range(epochs):
    layer1.forward(x)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    loss = loss_function.calculate(activation2.output, y)

    loss_function.backward(activation2.output, y)
    layer2.backward(loss_function.dinputs)
    activation1.backward(layer2.dinputs)
    layer1.backward(activation1.dinputs)

    layer1.weights -= learning_rate * layer1.dweights
    layer1.biases -= learning_rate * layer1.dbiases
    layer2.weights -= learning_rate * layer2.dweights
    layer2.biases -= learning_rate * layer2.dbiases

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} Total Loss: {loss:.4f}")

test_input = []
for i in range(0, 500):
    test_input.append([(rd.random() * 4 * math.pi)])
test_input = np.array(test_input)

layer1.forward(test_input)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)

loss = loss_function.calculate(activation2.output,  2* test_input)
prediction = activation2.output

print(loss)
print(np.sin(test_input[0]), prediction[0])
print(layer1.biases)
