import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Load data from CSV
data = pd.read_csv("C:\\Users\\Advik Narendran\\ml project\\patches_gabor_15816_1 3.csv")

# Separate features and target variable
X = data.drop(['class', 'ImageName'], axis=1)  # Drop both columns simultaneously
y = data['class']

# Encode target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define hyperparameter grid for RandomizedSearchCV
param_grid = {
    'hidden_layer_sizes': [(50,50,50), (100,), (50, 25, 10), (10,5)],  # Explore different architectures
    'activation': ['relu', 'tanh','sigmoid'],  # Try different activation functions
    'solver': ['adam', 'sgd'],  # Consider different optimizers
    'alpha': [0.0001, 0.001, 0.01],  # Test various regularization strengths
    'learning_rate_init': [0.001, 0.01],  # Experiment with initial learning rates
}

# Create RandomizedSearchCV object
random_search = RandomizedSearchCV(MLPClassifier(random_state=42), param_grid, n_iter=20, cv=5, verbose=2, random_state=42)

# Perform hyperparameter tuning
random_search.fit(X_train, y_train)

# Get best model and its parameters
best_model = random_search.best_estimator_
print("Best parameters:", random_search.best_params_)

# Evaluate best model on test data
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))
