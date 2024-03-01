import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Generate test set data
x_values = np.arange(0, 10.1, 0.1)
y_values = np.arange(0, 10.1, 0.1)
test_data = np.array(np.meshgrid(x_values, y_values)).T.reshape(-1, 2)

# Use the kNN classifier with k=3 (you'll need to replace this with your actual training data)
# Assuming you have X_train (training features) and y_train (training labels) from your model training
# Replace it with your actual data.
X_train = np.random.rand(100, 2) * 10
y_train = np.random.randint(2, size=100)

knn_classifier = KNeighborsClassifier(n_neighbors=16)
knn_classifier.fit(X_train, y_train)

# Predict the classes for the test data
predicted_classes = knn_classifier.predict(test_data)

# Scatter plot of the test data with points colored based on predicted classes
plt.scatter(test_data[:, 0], test_data[:, 1], c=predicted_classes, cmap='viridis')

# Add colorbar for better interpretation
plt.colorbar()

# Show the plot
plt.title('kNN Classifier - Test Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
