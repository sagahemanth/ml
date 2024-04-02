from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# Define the AND gate dataset
data = {
    "Input1": [0, 0, 1, 1],
    "Input2": [0, 1, 0, 1],
    "Output": [0, 0, 0, 1]
}

# Create a DataFrame from the dataset
df = pd.DataFrame(data)

# Split the dataset into features (X) and target variable (y)
X = df.drop(columns=["Output"])
y = df["Output"]

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define classifiers
classifiers = {
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0),
    "XGBoost": XGBClassifier(),
    "Naive Bayes": GaussianNB()
}

# Initialize a dictionary to store results
results = {"Classifier": [], "Accuracy": [], "Precision": [], "Recall": [], "F1 Score": [], "Error Percentage": []}

# Train and evaluate each classifier
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    error_percentage = (1 - accuracy) * 100
    
    # Append results to the dictionary
    results["Classifier"].append(name)
    results["Accuracy"].append(accuracy)
    results["Precision"].append(precision)
    results["Recall"].append(recall)
    results["F1 Score"].append(f1)
    results["Error Percentage"].append(error_percentage)

# Convert results to a pandas DataFrame
results_df = pd.DataFrame(results)

# Display the results
print(results_df)
