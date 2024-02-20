import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski

# Load your dataset from Excel
excel_file_path = "D:\ASEB\Semester 4\ML\patches_gabor_15816_1 3.csv"
df = pd.read_excel(excel_file_path)
print(df)

# Extract two feature vectors (assuming you have at least two rows in your dataset)
feature_vector1 = np.array(df.iloc[0, 1:])  # Adjust the row index accordingly
feature_vector2 = np.array(df.iloc[1, 2:])  # Adjust the row index accordingly

# Values of r from 1 to 10
r_values = np.arange(1, 11)

# Calculate Minkowski distances for each value of r
distances = [minkowski(feature_vector1, feature_vector2, p=r) for r in r_values]

# Plot the distances
plt.plot(r_values, distances, marker='o')
plt.title('Minkowski Distance vs. r')
plt.xlabel('r')
plt.ylabel('Minkowski Distance')
plt.show()
