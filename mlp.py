#!/usr/bin/python3
import numpy as np
import pickle

class Perceptron:
    def __init__(self, num_inputs):
        self.weights = np.random.uniform(-1, 1, num_inputs)
        self.bias = np.random.uniform(-1, 1)
        self.activation_function = self.threshold
        self.learning_rate = 0.1

    def threshold(self, x):
        return 1 if x > 0 else 0

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return self.activation_function(weighted_sum)

    def train(self, training_inputs, labels):
        for _ in range(1000):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def load(self, filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)

    def print_info(self):
        print(f"Perceptron : weights = {self.weights}, bias = {self.bias}, learning rate = {self.learning_rate}")

class MLP:
    def __init__(self, name_ia="ia", input_size=2, num_neurons_first_layer=3, num_hidden_layers=2, neurons_per_hidden_layer=4, output_neurons=1):
        self.name = name_ia
        self.layers = [num_neurons_first_layer] + [neurons_per_hidden_layer] * (num_hidden_layers) + [output_neurons]
        self.weights = [np.random.randn(y, x) for x, y in zip([input_size] + self.layers[:-1], self.layers)]
        self.biases = [np.random.randn(y, 1) for y in self.layers]
        self.activation_function = self.sigmoid
        self.learning_rate = 0.1

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def forward(self, x):
        activation = x
        activations = [x]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.activation_function(z)
            activations.append(activation)
        return zs, activations

    def backward(self, x, y, zs, activations):
        delta = (activations[-1] - y) * self.sigmoid_derivative(zs[-1])
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, len(self.layers) + 1):
            z = zs[-l]
            sp = self.sigmoid_derivative(z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)

        return nabla_b, nabla_w

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backward(x, y, *self.forward(x))
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def train(self, training_data, epochs, mini_batch_size, eta):
        n = len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

    def predict(self, x):
        activations = self.forward(x)[1]
        return activations[-1]

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.predict(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def save(self):
        with open(self.name, 'wb') as file:
            pickle.dump(self, file)

    def print_info(self):
        print("IA in file " + self.name)

        print("\nInput layer\t: " + "● " * self.layers[0])

        for i, layer_neurons in enumerate(self.layers[1:-1], start=1):
            print(f"Hidden layer {i}\t: " + "● " * layer_neurons)

        print("Output layer\t: " + "● " * self.layers[-1])
