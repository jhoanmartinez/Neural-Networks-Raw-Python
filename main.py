from neural_network import LayerDense, ReLU
from dataset import generate_dataset

# generate_data(samples_per_category, size_categories)
X_data, y_category = generate_dataset(3, 2)

print("Input data")
print(X_data)

layer1 = LayerDense(2, 4)

layer1.forward(X_data)

print("output")
print(layer1.output)

print("outout activated")
relu1 = ReLU()
relu1.forward(layer1.output)
print(relu1.output)