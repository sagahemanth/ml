import pandas as pd
import numpy as np
from math import log2

def entropy(data):
    """Calculate entropy of a dataset."""
    classes = data['class'].unique()
    entropy_val = 0
    total_instances = len(data)
    
    for c in classes:
        p = len(data[data['class'] == c]) / total_instances
        entropy_val -= p * log2(p)
        
    return entropy_val

def calculate_information_gain(data, feature):
    """Calculate information gain for a given feature."""
    unique_values = data[feature].unique()
    total_instances = len(data)
    feature_entropy = 0
    
    for val in unique_values:
        subset = data[data[feature] == val]
        weight = len(subset) / total_instances
        feature_entropy += weight * entropy(subset)
    
    return entropy(data) - feature_entropy

def find_root_node(data):
    """Find the feature with the highest information gain to be used as the root node."""
    # Exclude 'ImageName' and 'class' columns
    features = data.drop(columns=['ImageName', 'class']).columns
    max_info_gain = -1
    best_feature = None
    
    for feature in features:
        
        
        info_gain = calculate_information_gain(data, feature)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_feature = feature
    
    return best_feature

# Example usage:
# Load the dataset
data = pd.read_csv(r"D:\ASEB\Semester 4\ML\Lab8\patches_gabor_15816_1 3.csv")

# Find the root node feature
root_node_feature = find_root_node(data)
print("Root node feature:", root_node_feature)
