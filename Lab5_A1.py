import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


df = pd.read_csv("D:\ASEB\Semester 4\ML\patches_gabor_15816_1 3.csv")

X = df.iloc[:, 1:-1].values  # Exclude 'ImageName' and 'class' columns
y = df.iloc[:, -1].values

# Choose any two classes for binary classification
class_label1 = 'bad'
class_label2 = 'medium'

# Select rows corresponding to the chosen classes
class_data = df[(df['class'] == class_label1) | (df['class'] == class_label2)]

# Features and class labels for the chosen classes
X_selected = class_data.iloc[:, 1:-1].values
y_selected = class_data.iloc[:, -1].values

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_selected, test_size=0.3, random_state=42)

# Create a kNN classifier with k=3
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Train the classifier on the training set
knn_classifier.fit(X_train, y_train)

# Predictions on training set
y_train_pred = knn_classifier.predict(X_train)

# Predictions on test set
y_test_pred = knn_classifier.predict(X_test)

# Confusion matrix for training set
conf_matrix_train = confusion_matrix(y_train, y_train_pred)
print("Confusion Matrix (Training Set):")
print(conf_matrix_train)

# Confusion matrix for test set
conf_matrix_test = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix (Test Set):")
print(conf_matrix_test)

# Precision, Recall, and F1-Score for training set
precision_train = precision_score(y_train, y_train_pred, average='weighted')  # Use 'weighted' for multi-class
recall_train = recall_score(y_train, y_train_pred, average='weighted')
f1_score_train = f1_score(y_train, y_train_pred, average='weighted')

# Precision, Recall, and F1-Score for test set
precision_test = precision_score(y_test, y_test_pred, average='weighted')  # Use 'weighted' for multi-class
recall_test = recall_score(y_test, y_test_pred, average='weighted')
f1_score_test = f1_score(y_test, y_test_pred, average='weighted')

# Print performance metrics
print("\nPerformance Metrics (Training Set):")
print(f"Precision: {precision_train}")
print(f"Recall: {recall_train}")
print(f"F1-Score: {f1_score_train}")

print("\nPerformance Metrics (Test Set):")
print(f"Precision: {precision_test}")
print(f"Recall: {recall_test}")
print(f"F1-Score: {f1_score_test}")