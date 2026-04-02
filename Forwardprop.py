import numpy as np

# Input data
x = np.array([1, 2, 3, 4, 5, 6])
# Weights and bias
weights = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
bias = 0.5

# Activation function (ReLU)
def activation_function(x):
    return np.maximum(0, x)

# Forward propagation
def forward_propagation():
    # Calculate the weighted sum of inputs and weights
    weighted_sum = np.dot(x, weights) + bias
    # Apply the activation function
    output = activation_function(weighted_sum)
    return output

# Run the forward propagation
output = forward_propagation()
print("Output of the neural network:", output)