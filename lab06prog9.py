import numpy as np
import matplotlib.pyplot as plt

def perceptron(inputs, weights, activation):
    # Ensure inputs is a 2D array for the dot product operation
    inputs = np.atleast_2d(inputs)

    # Calculate the weighted sum (including bias)
    z = np.dot(weights, inputs.T) # Transpose inputs to match dimensions

    # Apply chosen activation function
    if activation == "step":
        output = 1 if z > 0 else 0 
    elif activation == "sigmoid":
        output = 1 / (1 + np.exp(-z))
    elif activation == "relu":
        output = max(0, z)
    elif activation == "bipolar_step":
        output = 1 if z > 0 else -1
    else:
        raise ValueError("Invalid activation function provided.")

    return output

def train_perceptron(data, target, epochs, learning_rate, initial_weights, activation):
  """
  Trains the perceptron model with a given learning rate and activation function.

  Args:
      data: A numpy array of training data points (each row is an input vector).
      target: A numpy array of target outputs.
      epochs: The number of training epochs.
      learning_rate: The learning rate for weight updates.
      initial_weights: A numpy array of initial weights (same dimension as data[0]).
      activation: A function representing the activation function.

  Returns:
      A tuple containing the final weights, bias (assumed to be zero), and a list of errors for each epoch.
  """
  weights = initial_weights[:len(data[0])]  # Take relevant weights based on input dimension
  errors = []

  for epoch in range(epochs):
    total_error = 0
    for i, (x, y) in enumerate(zip(data, target)):
      predicted = perceptron(x, weights, activation)
      error = y - predicted
      total_error += error**2

      # Update weights based on error
      weights += learning_rate * error * x

    # Calculate average error for the epoch
    average_error = total_error / len(data)
    errors.append(average_error)

    # Check for convergence
    if average_error <= 0.002:
      print(f"Converged in {epoch+1} epochs!")
      break

  return weights, 0.0, errors  # Assuming bias is zero

def plot_errors(epochs, errors, title):
  """
  Plots the errors vs epochs.

  Args:
      epochs: A list of epochs.
      errors: A list of errors for each epoch.
      title: The title for the plot.
  """
  plt.plot(epochs, errors)
  plt.xlabel("Epochs")
  plt.ylabel("Error")
  plt.title(title)
  plt.grid(True)
  plt.show()

# Define initial weights and bias (assuming bias is zero)
W0 = 10
W1 = 0.2
W2 = -0.75
bias = 0  # Assuming bias is zero
learning_rate = 0.05

# Sample training data (replace with your actual data)
data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target = np.array([0, 0, 0, 1])

activations = ["step", "sigmoid", "relu", "bipolar_step"]
for activation in activations:
  weights, bias, errors = train_perceptron(data, target, 1000, learning_rate, np.array([W0, W1, W2]), activation)
  print(f"Activation Function: {activation}")
  print(f"Final Weights: {weights}")
  print(f"Final Bias: {bias}")

  # New data point for prediction (replace with your actual data)
  new_data = np.array([0.2, 0.7])

  # Plot errors vs epochs
  plot_errors(range(1, len(errors)+1), errors, f"Perceptron Error ({activation})")