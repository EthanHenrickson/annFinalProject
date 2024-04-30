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

def normalize(values):
    min_value = np.min(values)
    max_value = np.max(values)

    normalized_values = (values - min_value) / (max_value - min_value)

    return normalized_values

x = []
y = []
for i in range(0,500):
    input1 = np.random.random()*10-5
    input2 = np.random.random()*10-5

    x.append([input1,input2])
    y.append([input1*input2])
x = np.array(x)
x= normalize(x)
y = np.array(y)


x = normalize(x)

# A layer with 1 input and 8 output neurons

hiddenNodes = 256
layer1 = Layer_Dense(2, hiddenNodes)

# Apply ReLU activation function to the output of the dense layer
activation1 = Activation_ReLU()

# Create a second layer with 8 input and 256 output neurons
layer2 = Layer_Dense(hiddenNodes, 128)

# Apply ReLU activation function to the output of the second dense layer
activation2 = Activation_ReLU()

# Create a third layer with 256 input and 1 output neurons
layer3 = Layer_Dense(128, 1)

# Calculate the loss using Mean Squared Error
loss_function = Loss_MeanSquaredError()

epochs = 5000
learning_rate = 0.005
for epoch in range(epochs):
    # Forward pass
    layer1.forward(x)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)
    layer3.forward(activation2.output)
    # Calculate loss
    loss = loss_function.calculate(layer3.output, y)

    if loss < .01:
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


og_input = []
test_input = []
a = []
b = []
c = []
y = []
for i in range(0,500):
    input1 = np.random.random()*10-5
    input2 = np.random.random()*10-5

    a.append(input1)
    b.append(input2)
    c.append(input1 * input2)
    y.append([input1*input2])
    og_input.append([input1,input2])
    test_input.append([input1,input2])


test_input = np.array(test_input)
test_input = normalize(test_input)

print(og_input[0])
print(y[0])
print("")
print("")
print("")


layer1.forward(test_input)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)
layer3.forward(activation2.output)

loss = loss_function.calculate(layer3.output, y)
prediction = layer3.output

print(prediction[0])
print(loss)

plt.plot(og_input, y)
plt.plot(test_input, prediction)

plt.show()
