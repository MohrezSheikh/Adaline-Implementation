import numpy as np

# Define the Adaline class
class Adaline:
    def __init__(self, learning_rate=0.1, max_epochs=100):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = None

    def activate(self, net_sum):
        # Activation function (Step function)
        if net_sum >= 0:
            return 1
        else:
            return 0

    def train(self, inputs, targets):
        # Randomly initialize weights and bias
        self.weights = np.random.rand(len(inputs[0]))
        self.bias = np.random.rand()

        for epoch in range(self.max_epochs):
            for i, x in enumerate(inputs):
                # Calculate the net input
                net_sum = np.dot(x, self.weights) + self.bias

                # Calculate the output
                output = self.activate(net_sum)

                # Calculate the error
                error = targets[i] - output

                # Update the weights and bias
                self.weights += self.learning_rate * error * x
                self.bias += self.learning_rate * error

    def predict(self, input):
        # Calculate the net input
        net_sum = np.dot(input, self.weights) + self.bias

        # Calculate the output
        output = self.activate(net_sum)

        return output


# Define the training data
inputs = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])

targets = np.array([0, 1, 1, 1])

# Create an instance of the Adaline class
adaline = Adaline(learning_rate=0.1, max_epochs=2)

# Train the Adaline network
adaline.train(inputs, targets)

# Test the trained network
for x in inputs:
    prediction = adaline.predict(x)
    print(f"Input: {x}, Prediction: {prediction}")