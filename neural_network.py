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
            values = softmax_row[range(len(y_pred)), true_categorie]
        elif len(true_categorie.shape) == 2:
            values = np.sum(softmax_row * true_categorie, axis=1)
        likelihood_loss = -np.log(values)
        mean_loss = np.mean(likelihood_loss)
        self.output = mean_loss
        return self.output
    
class Accuracy:

    def forward_accuracy(self, inputs_softmax, y_true):
        arg_max_softmax = np.argmax(inputs_softmax, axis=1)
        if len(y_true.shape) == 1:
            accuracy = np.mean(arg_max_softmax == y_true)
        elif len(y_true.shape) == 2:
            accuracy = np.mean(arg_max_softmax == y_true, axis=1)
        self.output = accuracy
        return self.output
        
