import numpy as np
from neural_network import LayerDense, ReLU
from dataset import generate_dataset, experiment_dataset

# Switch between dataset and a tiny dataset to check process and math
experiment = 1

if experiment == 1:
    # Tiny dataset to validate formulas and process
    X_data, y_category = experiment_dataset()
else:
    # Large dataset generate_data(samples_in_2dimension, number_of_Categories)
    X_data, y_category = generate_dataset(70, 3)

# Layer dense with two inputs and 3 output values
layer1 = LayerDense(2, 3)

# ReLU layer
relu1 = ReLU()

# Create forward pass
layer1.forward_layer(X_data)

# ReLU activation
relu1.forward_relu(layer1.output)







