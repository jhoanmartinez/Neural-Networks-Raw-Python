from neural_network import LayerDense
from dataset import generate_dataset

# Generate dataset
X_data, y_categories = generate_dataset(3, 3)

# Generate 1 dense layer
layer_1 = LayerDense(2, 3)

# Forward pass 1 dense layer
layer_1.forward(X_data)

# print values
print(layer_1.output)

