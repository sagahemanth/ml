import pandas as pd

# Your numerical data
data = [
    16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384,
    16384, 16384, 16384, 16384, 696575, 408825, 310949, 450988, 50610, 27865, 21379, 30316, 16384, 16384, 16384,
    16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 1411945, 770486, 533904, 552260, 489447,
    201353, 153389, 301361, 98604, 16384, 16384, 16384, 25214, 25734, 18661, 16384, 16384, 16384, 16384, 16384,
    16384, 16517, 16384, 20828, 18201, 16384, 19564, 16384, 16702, 16392, 16384, 16384, 16384, 16384, 16384,
    16384, 16384, 16384, 18559, 16384, 48319, 16384, 16440, 34882, 16384, 16384, 16384, 16384, 16384, 16765,
    19424, 28683, 33697, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 643793, 714961, 609861, 347397,
    828121, 174578, 78626, 56556, 17301, 16384, 17306
]

# Number of bins
num_bins = 3

# Calculate the range of values
data_range = max(data) - min(data)

# Calculate the width of each bin
bin_width = data_range / num_bins

# Create bins with equal widths
bins = [min(data) + i * bin_width for i in range(num_bins)]
bins.append(max(data) + 1)  # Add one more bin for values equal to the maximum

# Assign each value to its corresponding bin
binned_data = pd.cut(data, bins=bins, labels=False)

print(binned_data)
