import numpy as np

class LayerDense:

    def __init__(self, inputs_size, neuron_size):
        self.weights = 0.01 * np.random.rand(inputs_size, neuron_size)
        self.bias = np.zeros( (1, neuron_size) )

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias

class  ReLU:

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        