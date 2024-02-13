def find_pairs_with_sum(arr, target_sum):
    pairs = []
    seen_numbers = set()

    for num in arr:
        complement = target_sum - num
        if complement in seen_numbers:
            pairs.append((num, complement))
        seen_numbers.add(num)

    return pairs

# Example usage
given_list = [2, 7, 4, 1, 3, 6]
target_sum = 10
result_pairs = find_pairs_with_sum(given_list, target_sum)

print(f"Pairs with sum {target_sum}: {result_pairs}")
