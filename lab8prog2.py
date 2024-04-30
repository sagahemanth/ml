import pandas as pd
import numpy as np
from math import log2

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = {}

    def entropy(self, data):
        """Calculate entropy of a dataset."""
        classes = data['class'].unique()
        entropy_val = 0
        total_instances = len(data)
        
        for c in classes:
            p = len(data[data['class'] == c]) / total_instances
            entropy_val -= p * log2(p)
        
        return entropy_val

    def calculate_information_gain(self, data, feature):
        """Calculate information gain for a given feature."""
        unique_values = data[feature].unique()
        total_instances = len(data)
        feature_entropy = 0
        
        for val in unique_values:
            subset = data[data[feature] == val]
            weight = len(subset) / total_instances
            feature_entropy += weight * self.entropy(subset)
        
        return self.entropy(data) - feature_entropy

    def find_root_node(self, data):
        """Find the feature with the highest information gain to be used as the root node."""
        features = data.drop(columns=['ImageName', 'class']).columns
        max_info_gain = -1
        best_feature = None
        
        for feature in features:
            info_gain = self.calculate_information_gain(data, feature)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_feature = feature
        
        return best_feature

    def binning(self, data, num_bins=None, binning_type='equal_width'):
        """
        Bins continuous-valued attributes into categorical-valued attributes using equal width or frequency binning.
        """
        if num_bins is None:
            num_bins = int(len(data) ** 0.5)

        if binning_type == 'equal_width':
            for column in data.columns:
                if data[column].dtype != 'object':
                    data[column] = pd.cut(data[column], bins=num_bins, labels=False)
        elif binning_type == 'frequency':
            for column in data.columns:
                if data[column].dtype != 'object':
                    data[column] = pd.qcut(data[column], q=num_bins, labels=False, duplicates='drop')

        return data

    def fit(self, data):
        """Fit the decision tree to the data."""
        # Binning the dataset
        data_binned = self.binning(data)
        
        # Finding the root node
        root_node_feature = self.find_root_node(data_binned)
        
        # Recursive splitting logic
        self.tree = self._build_tree(data_binned, root_node_feature, 0)

    def _build_tree(self, data, feature, depth):
        """Recursively build the decision tree."""
        if depth == self.max_depth or len(data['class'].unique()) == 1:
            return data['class'].mode()[0]
        
        info_gain = self.calculate_information_gain(data, feature)
        if info_gain <= 0:
            return data['class'].mode()[0]
        
        tree = {}
        for value in data[feature].unique():
            subset = data[data[feature] == value]
            if not subset.empty:
                best_feature = self.find_root_node(subset)
                tree[value] = self._build_tree(subset, best_feature, depth + 1)
        
        return tree

    def predict(self, data):
        """Predict the class for a given data point."""
        # Ensure data is a DataFrame
        if isinstance(data, pd.Series):
            data = data.to_frame().T
        return self._predict(self.tree, data)

    def _predict(self, tree, data):
        """Recursively traverse the tree to make a prediction."""
        if isinstance(tree, str):
            return tree
        
        feature_value = data.iloc[0][list(tree.keys())[0]] # Corrected to use .iloc for positional access
        if feature_value in tree:
            return self._predict(tree[feature_value], data)
        else:
            # If the feature value is not in the tree, return the most common class
            # Assuming 'data' is a DataFrame for the root prediction
            return data['class'].mode()[0]

# Example usage:
# Load the dataset
data = pd.read_csv(r"D:\ASEB\Semester 4\ML\Lab8\patches_gabor_15816_1 3.csv")

# Create and fit the decision tree
tree = DecisionTree(max_depth=3)
tree.fit(data)

# Make a prediction
prediction = tree.predict(data.iloc[0])
print("Prediction:", prediction)
