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


class Loss_MeanSquaredError():
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


x = np.random.randn(500,2)*10 -5
y = []
for item in x:
     y.append([item[0]*item[1]])

# A layer with 1 input and 8 output neurons

hiddenNodes = 512

layer1 = Layer_Dense(2, hiddenNodes)
# Apply ReLU activation function to the output of the dense layer
activation1 = Activation_ReLU()
# Create a second layer with 8 input and 1 output neurons
layer2 = Layer_Dense(hiddenNodes, 1)
# Calculate the loss using Mean Squared Error
loss_function = Loss_MeanSquaredError()
epochs = 15000
learning_rate = 0.0075
for epoch in range(epochs):
    # Forward pass
    layer1.forward(x)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    # Calculate loss
    loss = loss_function.calculate(layer2.output, y)

    

    loss_function.backward(layer2.output, y)
    layer2.backward(loss_function.dinputs)
    activation1.backward(layer2.dinputs)
    layer1.backward(activation1.dinputs)
    # Update weights and biases
    layer1.weights -= learning_rate * layer1.dweights
    layer1.biases -= learning_rate * layer1.dbiases
    layer2.weights -= learning_rate * layer2.dweights
    layer2.biases -= learning_rate * layer2.dbiases
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} Total Loss: {loss:.4f}")
# Test the trained model


test_input = []
for i in range(0,500):
	test_input.append([12/500*i])
test_input = np.array(test_input)

layer1.forward(test_input)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
loss = loss_function.calculate(layer2.output, np.sin(test_input*2))
prediction = layer2.output

plt.plot(test_input, np.sin(test_input * 2))
plt.plot(test_input, prediction)


plt.show()

print(loss)
