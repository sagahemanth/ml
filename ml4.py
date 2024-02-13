def max_occurrence(input_string):
    char_count = {}
    for char in input_string:
        if char.isalpha():
            char_count[char] = char_count.get(char, 0) + 1
    max_char = max(char_count, key=char_count.get)
    max_count = char_count[max_char]
    return max_char, max_count

input_str = "hippopotamus"
max_char, max_count = max_occurrence(input_str)
print(f"The maximally occurring character is '{max_char}' with occurrence count {max_count}.")

