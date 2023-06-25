import numpy as np

class LayerDense:

    def __init__(self, inputs, neuron) -> None:
        self.weights = 0.01 * np.random.rand(inputs, neuron)
        self.bias = np.zeros( (1, neuron) )

    def forward(self, input):
        self.output = np.dot(input, self.weights) + self.bias

        