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


class BinaryCrossEntropyLoss:

    def forward_BCEL(self, y_pred, y_true):
        softmax_row = y_pred # [ [0.3, 0.3], [0.4,0.4] ]
        true_categorie = y_true # [0,0,1,1,2,2]
        if len(true_categorie.shape) == 1:
            body = softmax_row[range(len(y_pred)), true_categorie]
        elif (true_categorie.shape) == 2:
            body = np.sum(softmax_row * true_categorie)
        
        self.output = body
        
