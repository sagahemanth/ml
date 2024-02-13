def onehot_encoding(categories):
    unique_categories = list(set(categories))
    encoding = {}
    for i, category in enumerate(unique_categories):
        encoding[category] = [0] * i + [1] + [0] * (len(unique_categories) - i - 1)
    return [encoding[category] for category in categories]

categories = ['One', 'Two', 'One', 'Three', 'Three', 'One', 'Two']
encoded_values = onehot_encoding(categories)
print("One-hot encoded values are:")
for category, encoded in zip(categories, encoded_values):
    print(f"{category}: {encoded}")
