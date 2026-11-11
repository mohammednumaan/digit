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
                print(f"Epoch {j} complete")


    def update_mini_batch(self, mini_batch, eta):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) 
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]    
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]    


    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        activation = x
        activations = [x]
        z_arr = []

        # the first thing i need to do is 'feedforward'
        for b, w in zip(self.biases, self.weights):
            # this is simple. it does matrix multiplication
            # with the weight matrix w and activation vectors a
            # and finally does matrix addition with the bias vector
            # so lets say we have 2 neurons so z would give us [val1, val2]  
            z = np.dot(w, activation) + b
            z_arr.append(z)

            # this basically does "vectorization"
            # which means it applies the sigmoid function elementwise
            # i.e to each element in the z vector
            activation = sigmoid(z)
            activations.append(activation)

        # the next thing i need to do is 'backprop'
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(z_arr[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) 

        # now i can start backpropagating to the hidden layers
        # here l is the "distance" from the output layer and not the
        # actual layer number. so if l = 2 it means its the second last layer 
        for l in range(2, self.num_layers):
            z = z_arr[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose()) 

        return (nabla_b, nabla_w)

        

    def cost_derivative(self, output_activation, y):
        # this is basically what we get if we diffrentiate
        # the cost function 1/2 * (y  - a)^2
        # its essentially: delC_x/dela
        return (output_activation - y)
    
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

