import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# Define functions for AND and XOR gates
def AND(x1, x2):
  return np.bitwise_and(x1, x2)

def XOR(x1, x2):
  return np.bitwise_xor(x1, x2)

# Generate data for AND gate
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = AND(X_and[:, 0], X_and[:, 1])

# Generate data for XOR gate
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = XOR(X_xor[:, 0], X_xor[:, 1])

# Define function to train and plot errors
def train_and_plot(X, y, title):
  errors = []
  for epoch in range(1000):
    # Define and train the MLP classifier
    mlp = MLPClassifier(activation='relu', solver='lbfgs', hidden_layer_sizes=(4,), max_iter=1)
    mlp.fit(X, y)

    # Predict and calculate error
    y_pred = mlp.predict(X)
    error = np.mean((y - y_pred)**2)
    errors.append(error)

    # Stop if convergence criteria met
    if error <= 0.002:
      break

  # Print predicted output
  print(f"\nPredicted Outputs for {title} Gate:")
  for i in range(len(X)):
    print(f"Input: {X[i]}, Predicted: {y_pred[i]}")

  # Plot epochs vs error
  plt.plot(range(len(errors)), errors)
  plt.xlabel("Epochs")
  plt.ylabel("Mean Squared Error")
  plt.title(f"{title} Gate Error")
  plt.show()

# Train and plot for AND gate
train_and_plot(X_and, y_and, "AND")

# Train and plot for XOR gate
train_and_plot(X_xor, y_xor, "XOR")
