from collections import Counter

def select_majority_values(lists):
    # Transpose the list of lists to group elements by their positions
    transposed = list(zip(*lists))
    
    # Find the majority element for each group
    majority_values = []
    for group in transposed:
        # Get the most common element
        most_common = Counter(group).most_common(1)[0][0]
        majority_values.append(most_common)
    
    return majority_values

# Example usage
input_data = [[1, 0, 0, 1, 3], [0, 1, 1, 1, 3], [1, 0, 1, 1, 1]]
result = select_majority_values(input_data)
print(result)  # Output: [0, 0, 1, 1, 1]