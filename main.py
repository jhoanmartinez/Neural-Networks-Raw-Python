import numpy as np
from neural_network import LayerDense, ReLU, Softmax
from dataset import generate_dataset


# Large dataset generate_data(samples_in_2dimension, number_of_Categories)
X_data, y_category = generate_dataset(3, 3)

""" ======>>>>>>>   Layers architecture <<<<<<======"""

# Layer dense with two inputs and 3 output values
layer1 = LayerDense(2, 3)

# ReLU layer
relu1 = ReLU()

# Softmax 1
softmax1 = Softmax()

# Create forward pass
layer1.forward_layer(X_data)
print("forward pass =\n",layer1.output)

# ReLU activation
relu1.forward_relu(layer1.output)
print("\nrelu activation =\n", relu1.output)

# Softmax activation
softmax1.forward_softmax(relu1.output)
print("\nSoftmax activation =\n", softmax1.output)







