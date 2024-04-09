import pandas as pd
import numpy as np

def parse_data_string(data_string):
    # Remove 'PS: ' and square brackets, then split the string by comma and space
    values = data_string.replace('PS: [', '').replace(']', '').split(',')
    # Convert each pair of values to a tuple of floats
    data_tuples = [(float(values[i]), float(values[i+1])) for i in range(0, len(values), 2)]
    # Convert tuples to a flat array
    return np.array(data_tuples).flatten()

# Example usage:
data_string = "PS: [0.01,-0.01, -0.00,-0.00, -0.00,-0.00,-0.00,0.01]"
data_array = parse_data_string(data_string)
print(data_array)
