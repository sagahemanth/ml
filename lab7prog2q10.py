from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
import numpy as np

# Define input and output for AND gate
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

# Define input and output for XOR gate
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# Define the MLPClassifier
mlp = MLPClassifier(max_iter=1000)

# Define hyperparameters to search
param_grid = {
    'hidden_layer_sizes': [(4,), (8,), (16,), (32,)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive']
}

# For AND gate
print("Tuning hyperparameters for AND gate:")
and_search = RandomizedSearchCV(mlp, param_distributions=param_grid, n_iter=10, cv=3)
and_search.fit(X_and, y_and)

print("Best parameters found for AND gate:")
print(and_search.best_params_)
print("Best score found for AND gate:", and_search.best_score_)

# For XOR gate
print("\nTuning hyperparameters for XOR gate:")
xor_search = RandomizedSearchCV(mlp, param_distributions=param_grid, n_iter=10, cv=2)
xor_search.fit(X_xor, y_xor)

print("Best parameters found for XOR gate:")
print(xor_search.best_params_)
print("Best score found for XOR gate:", xor_search.best_score_)
