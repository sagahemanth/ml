import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  # Setting seed for reproducibility
num_points = 20

X = np.random.uniform(1, 10, num_points)
Y = np.random.uniform(1, 10, num_points)
# Example: Assign class0 (Blue) if X is greater than the mean, else assign class1 (Red)
classes = np.where(X > np.mean(X), 1, 0)
# Color mapping for classes
colors = ['blue' if c == 0 else 'red' for c in classes]

# Create scatter plot
plt.scatter(X, Y, c=colors)
plt.title('Scatter Plot of Training Data')
plt.xlabel('Feature X')
plt.ylabel('Feature Y')
plt.show()
