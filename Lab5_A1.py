import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Load your dataset
dataset = pd.read_excel("C:\Users\heman\OneDrive\Documents\patches_gabor_15816_1 3 (1).csv")

# Extract features (X) and target variable (y)
X = dataset.drop(columns=["output"]).values  # Features
y = dataset["output"].values  # Target variable
y = pd.cut(y, bins=3, labels=["low", "medium", "high"]) #Making y values discrete 

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

neigh=KNeighborsClassifier(n_neighbors=3)
neigh.fit(X,y)

accuracy = neigh.score(X_test,y_test)
print("NN Accuracy (k = 3):",accuracy)

y_pred = neigh.predict(X_test)
test_index = 0  # Index of the test vector 
test_vect = X_test[test_index]

# Perform classification for the given test vector
predicted_class = neigh.predict([test_vect])
print("Predicted class for the test vector for NN accuracy (k = 3):", predicted_class)

# Predict classes for the training and test data using both classifiers
nn_train_predictions = neigh.predict(X_train)
nn_test_predictions = neigh.predict(X_test)

# Calculate confusion matrix and classification report for Nearest Neighbor (NN) classifier
nn_train_conf_matrix = confusion_matrix(y_train, nn_train_predictions)
nn_test_conf_matrix = confusion_matrix(y_test, nn_test_predictions)
nn_train_report = classification_report(y_train, nn_train_predictions)
nn_test_report = classification_report(y_test, nn_test_predictions)

# Print confusion matrices and classification reports
print("Confusion Matrix for K Nearest Neighbor (k=3) Classifier (Training Data):\n", nn_train_conf_matrix)
print("\nConfusion Matrix for K Nearest Neighbor (k=3) Classifier (Test Data):\n", nn_test_conf_matrix)
print("\nClassification Report for K Nearest Neighbor (k=3) Classifier (Training Data):\n", nn_train_report)
print("\nClassification Report for K Nearest Neighbor (k=3) Classifier (Test Data):\n", nn_test_report)
