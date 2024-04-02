from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import randint

# Load dataset
X, y = load_digits(return_X_y=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the perceptron model
perceptron = Perceptron()

# Define hyperparameters to tune for perceptron
param_dist_perceptron = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    'max_iter': randint(100, 1000),
    'tol': [1e-3, 1e-4, 1e-5],
    'eta0': [0.1, 0.01, 0.001],
    'random_state': [42]
}

# Perform RandomizedSearchCV for perceptron
perceptron_search = RandomizedSearchCV(perceptron, param_distributions=param_dist_perceptron, n_iter=100, cv=5, scoring='accuracy', random_state=42)
perceptron_search.fit(X_train, y_train)

# Get the best parameters for perceptron
best_params_perceptron = perceptron_search.best_params_

# Train the perceptron with the best parameters
best_perceptron = Perceptron(**best_params_perceptron)
best_perceptron.fit(X_train, y_train)

# Evaluate perceptron on test set
perceptron_predictions = best_perceptron.predict(X_test)
perceptron_accuracy = accuracy_score(y_test, perceptron_predictions)
print("Perceptron Accuracy:", perceptron_accuracy)

# Define the MLP model
mlp = MLPClassifier()

# Define hyperparameters to tune for MLP
param_dist_mlp = {
    'hidden_layer_sizes': [(50,),(100,),(150,),(200,)],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'max_iter': randint(100, 1000),
    'tol': [1e-3, 1e-4, 1e-5],
    'random_state': [42]
}

# Perform RandomizedSearchCV for MLP
mlp_search = RandomizedSearchCV(mlp, param_distributions=param_dist_mlp, n_iter=100, cv=5, scoring='accuracy', random_state=42)
mlp_search.fit(X_train, y_train)

# Get the best parameters for MLP
best_params_mlp = mlp_search.best_params_

# Train MLP with the best parameters
best_mlp = MLPClassifier(**best_params_mlp)
best_mlp.fit(X_train, y_train)

# Evaluate MLP on test set
mlp_predictions = best_mlp.predict(X_test)
mlp_accuracy = accuracy_score(y_test, mlp_predictions)
print("MLP Accuracy:", mlp_accuracy)
