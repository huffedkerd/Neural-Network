# Backward Propagation
import numpy as np
# Input data
x = np.array([1, 2, 3, 4, 5, 6])
# Weights and bias
weights = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
bias = 0.5
# Target output (for example, we want the output to be 10)
target_output = 10
# Activation function (ReLU)
def activation_function(x):
    return np.maximum(0, x)
# Derivative of the activation function (ReLU)
def activation_function_derivative(x):
    return np.where(x > 0, 1, 0)
# Forward propagation
def forward_propagation():
    # Calculate the weighted sum of inputs and weights
    weighted_sum = np.dot(x, weights) + bias
    # Apply the activation function
    output = activation_function(weighted_sum)
    return output, weighted_sum
# Backward propagation
def backward_propagation(output, weighted_sum):
    # Calculate the error
    error = target_output - output
    # Calculate the derivative of the output with respect to the weighted sum
    d_output_d_weighted_sum = activation_function_derivative(weighted_sum)
    # Calculate the gradient of the error with respect to the weights and bias
    d_error_d_weighted_sum = -error * d_output_d_weighted_sum
    d_error_d_weights = d_error_d_weighted_sum * x
    d_error_d_bias = d_error_d_weighted_sum
    return d_error_d_weights, d_error_d_bias
# Run the forward propagation
output, weighted_sum = forward_propagation()
print("Output of the neural network:", output)
# Run the backward propagation
d_error_d_weights, d_error_d_bias = backward_propagation(output, weighted_sum)
print("Gradient of the error with respect to weights:", d_error_d_weights)
print("Gradient of the error with respect to bias:", d_error_d_bias)