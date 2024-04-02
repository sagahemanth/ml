import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings


# Load data from CSV
data = pd.read_csv("C:\\Users\\Advik Narendran\\ml project\\gabormin4.csv")

# Separate features and target variable
X = data.drop(['class', 'ImageName'], axis=1)
y = data['class']

# Encode target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define hyperparameter grid for RandomizedSearchCV for MLPClassifier
param_grid_mlp = {
    'hidden_layer_sizes': [(50,50,50), (100,), (50, 25, 10), (10,5)],
    'activation': ['relu', 'tanh','logistic'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01],
}

# Define hyperparameter grid for RandomizedSearchCV for SVM
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
}

# Define hyperparameter grid for RandomizedSearchCV for Decision Tree
param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40, 50],
}

# Define hyperparameter grid for RandomizedSearchCV for RandomForest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30, 40, 50],
}

# Define hyperparameter grid for RandomizedSearchCV for AdaBoost
param_grid_adaboost = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1],
}

# Define hyperparameter grid for RandomizedSearchCV for XGBoost
param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.1, 0.2],
}

# Define hyperparameter grid for RandomizedSearchCV for CatBoost
param_grid_catboost = {
    'iterations': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
}

# Define hyperparameter grid for RandomizedSearchCV for Naive Bayes (no hyperparameters to tune)
param_grid_nb = {}

# Define classifiers
classifiers = {
    'MLPClassifier': (MLPClassifier(random_state=42), param_grid_mlp),
    'SVM': (SVC(random_state=42), param_grid_svm),
    'Decision Tree': (DecisionTreeClassifier(random_state=42), param_grid_dt),
    'Random Forest': (RandomForestClassifier(random_state=42), param_grid_rf),
    'AdaBoost': (AdaBoostClassifier(random_state=42), param_grid_adaboost),
    'XGBoost': (XGBClassifier(random_state=42), param_grid_xgb),
    'CatBoost': (CatBoostClassifier(random_state=42, verbose=0), param_grid_catboost),
    'Naive Bayes': (GaussianNB(), param_grid_nb),
}

# Performance metrics
metrics = {
    'Accuracy': accuracy_score,
    'Precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro'),
    'Recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro'),
    'F1 Score': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
}

results = []
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# Iterate over classifiers
for clf_name, (clf, param_grid) in classifiers.items():
    print(f"Training {clf_name}...")
    # Create RandomizedSearchCV object
    random_search = RandomizedSearchCV(clf, param_grid, n_iter=5, cv=5, verbose=0, random_state=42)
    # Perform hyperparameter tuning
    random_search.fit(X_train, y_train)
    # Get best model and its parameters
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    # Evaluate best model on test data
    y_pred = best_model.predict(X_test)
    # Calculate performance metrics
    clf_results = {'Classifier': clf_name}
    for metric_name, metric_func in metrics.items():
        clf_results[metric_name] = metric_func(y_test, y_pred)
    # Append results
    results.append(clf_results)

# Create DataFrame for results
results_df = pd.DataFrame(results)
print(results_df)
