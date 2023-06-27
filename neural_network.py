import numpy as np

class LayerDense:

    def __init__(self, inputs_size, neuron_size):
        # self.weights = 0.01 * np.random.rand(inputs_size, neuron_size)
        self.weights = [
                        [1.1, 1.2, 1.3],
                        [1.2, 1.2, 1.3]
                    ]
        self.bias = np.zeros( (1, neuron_size) )

    def forward_layer(self, inputs):
        inputs = [
        [1, 2],
        [7, 8],
        [2, 9]
    ]
        self.output = np.dot(inputs, self.weights) + self.bias

class  ReLU:

    def forward_relu(self, inputs):
        self.output = np.maximum(10, inputs)

class Softmax:

    def forward(self, inputs):
        pass


        