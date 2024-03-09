import math

def sigmoid(x):
  """Sigmoid activation function."""
  return 1 / (1 + math.exp(-x))

def perceptron(inputs, weights):
  """Perceptron with sigmoid activation."""
  z = w0 + inputs[0] * weights[1] + inputs[1] * weights[2]
  return sigmoid(z)

def train_perceptron(inputs, targets, weights, learning_rate):
  """Train perceptron with sigmoid activation."""
  for i in range(len(inputs)):
    output = perceptron(inputs[i], weights)
    error = targets[i] - output
    # Update weights using derivative of sigmoid
    for j in range(len(weights)):
      weights[j] += learning_rate * error * output * (1 - output) * inputs[i][j - 1]

w0 = 0  # Initialize weights to avoid bias
w1 = 0.2
w2 = -0.75
learning_rate = 0.1  # May require adjustments

weights = (w0, w1, w2)  # Use tuple for weights

inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
targets = [0, 1, 1, 0]

epochs = 1000  # May require more epochs for sigmoid

for epoch in range(epochs):
  train_perceptron(inputs, targets, weights, learning_rate)

print("Learned weights:", weights)

# Test the learned perceptron
for input, target in zip(inputs, targets):
  output = perceptron(input, weights)
  print("Input:", input, "Target:", target, "Output:", output)
