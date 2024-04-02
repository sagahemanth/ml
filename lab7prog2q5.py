from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Prepare the data
data = np.array([
    [20, 6, 2, 1],
    [16, 3, 6, 1],
    [27, 6, 2, 1],
    [19, 1, 2, 0],
    [24, 4, 2, 1],
    [22, 1, 5, 0],
    [15, 4, 2, 1],
    [18, 4, 2, 1],
    [21, 1, 4, 0],
    [16, 2, 4, 0]
])

# Normalize the features
scaler = StandardScaler()
data[:, :-1] = scaler.fit_transform(data[:, :-1])

X = data[:, :-1]
y = data[:, -1]

# Define hyperparameter grid for RandomizedSearchCV
param_grid = {
    "max_iter": np.arange(100, 1000, step=1),  # Wider range
    "tol": np.logspace(-5, -1, base=10, num=5),  # Logarithmic scale
    "eta0": np.linspace(0.005, 0.15, 15)  # More values and finer range
}

# Create a Perceptron classifier instance
clf = Perceptron()

# Create RandomizedSearchCV object
randomized_search = RandomizedSearchCV(
    estimator=clf, param_distributions=param_grid, cv=5, n_iter=20, random_state=0
)

# Perform hyperparameter tuning
randomized_search.fit(X, y)

# Get the best model and its parameters
best_model = randomized_search.best_estimator_
best_params = randomized_search.best_params_

# Print best parameters
print("Best parameters:", best_params)

# Make predictions using the best model
predictions = best_model.predict(X)

# Apply sigmoid function for classification
sigmoid_output = 1 / (1 + np.exp(-predictions))
high_value_predictions = [1 if i > 0.5 else 0 for i in sigmoid_output]

print("Predictions:", high_value_predictions)
print("Actual:", y)
