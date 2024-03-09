import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

# Load data from CSV (replace 'your_data.csv' with your actual file path)
data = pd.read_csv("C:\\Users\\Advik Narendran\\ml project\\patches_gabor_15816_1 3.csv")

# Separate features and target variable
X = data.drop('class', axis=1)  
X= X.drop('ImageName',axis=1)# Replace 'target_column' with your actual column name
y = data['class']

# Encode target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Preprocess data (assuming numerical features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the MLP model
mlp = MLPClassifier(solver='adam', alpha=0.001, hidden_layer_sizes=(10, 5), random_state=42)

# Train the model
mlp.fit(X_train, y_train)

# Make predictions on test data
y_pred = mlp.predict(X_test)
# Evaluate model performance (e.g., using classification report)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# For in-memory prediction on new data:
#  1. Preprocess new data using the scaler
#  2. Use mlp.predict(new_data) to get predictions

# This example is for in-memory prediction. Database integration requires further development specific to your database system.
