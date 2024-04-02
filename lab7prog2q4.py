from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import numpy as np

# Define the activation functions
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def bipolar_step(x):
    return np.where(x >= 0, 1, -1)

class Perceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, activation='relu', alpha=0.05, max_iter=1000):
        self.activation = activation
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(self, X, y):
        self.weights = np.random.rand(X.shape[1])
        self.bias = 10
        for _ in range(self.max_iter):
            if self.activation == 'relu':
                activations = relu(np.dot(X, self.weights) + self.bias)
            elif self.activation == 'sigmoid':
                activations = sigmoid(np.dot(X, self.weights) + self.bias)
            elif self.activation == 'bipolar_step':
                activations = bipolar_step(np.dot(X, self.weights) + self.bias)
            
            error = y - activations
            self.weights += self.alpha * np.dot(X.T, error)
            self.bias += self.alpha * np.sum(error)
        return self

    def predict(self, X):
        if self.activation == 'relu':
            activations = relu(np.dot(X, self.weights) + self.bias)
        elif self.activation == 'sigmoid':
            activations = sigmoid(np.dot(X, self.weights) + self.bias)
        elif self.activation == 'bipolar_step':
            activations = bipolar_step(np.dot(X, self.weights) + self.bias)
        return np.where(activations >= 0.5, 1, 0)



# Define the input data and labels for the XOR gate

# Define the parameter grid for RandomizedSearchCV for each activation function
param_grid_relu = {
    'alpha': [0.01, 0.05, 0.1, 0.5, 1],
    'max_iter': [100, 500, 1000, 2000]
}

param_grid_sigmoid = {
    'alpha': [0.01, 0.05, 0.1, 0.5, 1],
    'max_iter': [100, 500, 1000, 2000]
}

param_grid_bipolar_step = {
    'alpha': [0.01, 0.05, 0.1, 0.5, 1],
    'max_iter': [100, 500, 1000, 2000]
}


X_xor = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

y_xor = np.array([0, 1, 1, 0])

# Perform RandomizedSearchCV for each activation function
random_search_relu_xor = RandomizedSearchCV(Perceptron(activation='relu'), param_distributions=param_grid_relu, n_iter=20, cv=2)
random_search_sigmoid_xor = RandomizedSearchCV(Perceptron(activation='sigmoid'), param_distributions=param_grid_sigmoid, n_iter=20, cv=2)
random_search_bipolar_step_xor = RandomizedSearchCV(Perceptron(activation='bipolar_step'), param_distributions=param_grid_bipolar_step, n_iter=20, cv=2)

# Fit RandomizedSearchCV for each activation function
random_search_relu_xor.fit(X_xor, y_xor)
random_search_sigmoid_xor.fit(X_xor, y_xor)
random_search_bipolar_step_xor.fit(X_xor, y_xor)

# Get the best parameters for each activation function
best_params_relu_xor = random_search_relu_xor.best_params_
best_params_sigmoid_xor = random_search_sigmoid_xor.best_params_
best_params_bipolar_step_xor = random_search_bipolar_step_xor.best_params_

print("Best Parameters for ReLU (XOR):", best_params_relu_xor)
print("Best Parameters for Sigmoid (XOR):", best_params_sigmoid_xor)
print("Best Parameters for Bipolar Step (XOR):", best_params_bipolar_step_xor)
