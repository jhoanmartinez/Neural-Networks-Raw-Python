import numpy as np

class LayerDense:

    def __init__(self, inputs_size, neuron_size):
        self.weights = 0.01 * np.random.rand(inputs_size, neuron_size)
        self.bias = np.zeros( (1, neuron_size) )

    def forward_layer(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias

class  ReLU:

    def forward_relu(self, inputs):
        self.output = np.maximum(0, inputs)

class Softmax:

    def forward_softmax(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        distribution_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = distribution_values


        