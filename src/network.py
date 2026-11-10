import random
import numpy as np

class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    def sgd(self, training_data, epochs, mini_batch_size, eta, test_data = None):

        n_test = 0
        if test_data:
            n_test = len(test_data)

        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            for mb in mini_batches:
                self.update_mini_batch(mb, eta)

            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {0} complete".format(j))


    def update_mini_batch(self, mini_batch, eta):
        pass

    def evaluate(self, test_data):
        pass


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))


net = Network([2, 3, 1])
print(net.weights)
print(net.biases)
