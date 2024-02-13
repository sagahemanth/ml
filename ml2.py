def calculate_range(numbers):
    if len(numbers) < 3:
        return "Range determination not possible"

    return max(numbers) - min(numbers)

# Example usage
input_list = [5, 3, 8, 1, 0, 4]
result_range = calculate_range(input_list)

print(f"Range of the list: {result_range}")

print (f"Range of the list: {result_range}")