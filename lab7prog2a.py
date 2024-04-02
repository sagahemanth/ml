import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Define the Perceptron function for clarity
def perceptron_model(X, y, weights):
    predictions = np.dot(X, weights)
    outputs = np.where(predictions >= 0, 1, 0)
    return outputs

# Constants
w0 = 10
w1 = 0.2
w2 = -0.75
a = 0.05  # Initial learning rate

# Inputs and target output for AND gate
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target_output = np.array([0, 0, 0, 1])

# Define hyperparameter search space
param_distributions = {
    'eta0': [0.001, 0.01, 0.1],  # Learning rate
    'tol': [1e-4, 1e-3, 1e-2],    # Tolerance
    'max_iter': [50, 100, 200]   # Maximum iterations
}

# Create the Perceptron instance
perceptron = Perceptron(random_state=42)

# Employ RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(
    estimator=perceptron,
    param_distributions=param_distributions,
    n_iter=20,
    scoring='accuracy',
    cv=2  # Adjust cv as needed
)

# Add bias term to inputs for compatibility with Perceptron class
inputs_with_bias = np.insert(inputs, 0, 1, axis=1)

# Fit the RandomizedSearchCV
random_search.fit(inputs_with_bias, target_output)

# Retrieve the best hyperparameters
best_params = random_search.best_params_
print("Best hyperparameters:", best_params)

# Create the model with the best hyperparameters
best_model = Perceptron(
    eta0=best_params['eta0'],
    tol=best_params['tol'],
    max_iter=best_params['max_iter']
)

# Train the model with the best hyperparameters
best_model.fit(inputs_with_bias, target_output)

# Get the learned weights
# Learned weights and predictions (assuming successful fit)
learned_weights = best_model.coef_.ravel()
predictions = best_model.predict(inputs_with_bias)

# Evaluate model performance (adjust metric as needed)
# Evaluate model performance (adjust metric as needed)
accuracy = accuracy_score(target_output, predictions)
print("Predictions:", predictions)
print("Accuracy:", accuracy)
accuracy = accuracy_score(target_output, predictions)
print("Accuracy:", accuracy)