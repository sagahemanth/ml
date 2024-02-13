def categorical_to_numerical(categories):
    encoding_val = {}
    count = 0
    for i in categories:
        if i not in encoding_val:
            encoding_val[i] = count
            count += 1
    return [encoding_val[i] for i in categories]


categories = ['One', 'Two', 'One', 'Three', 'Three', 'One', 'Two']
encoded_values = categorical_to_numerical(categories)
print("Encoded values:", encoded_values)
