# transforms.py
import numpy as np

def scale_to_0_and_1(data_list, upper_range=1):
    """
    Scale the input data list to the range [0, upper_range].

    Parameters:
        data_list (list): List of numerical values.
        upper_range (float): The upper bound of the scaling range (default is 1).

    Returns:
        list: List of values scaled to the specified range.
    """
    # Check if data_list is a non-empty list
    if not isinstance(data_list, list) or len(data_list) == 0:
        raise ValueError("data_list must be a non-empty list")
    # Check if upper_range is a number
    if not isinstance(upper_range, (int, float)):
        raise TypeError("upper_range must be an int or float")
        
    # Replace negative values with zero
    data_list = [0 if i < 0 else i for i in data_list]
    min_value = min(data_list)
    max_value = max(data_list)
    range_value = max_value - min_value
    # If all values are the same, avoid division by zero by returning zeros
    if range_value == 0:
        return [0 for _ in data_list]
    # Scale each element to the specified upper range and return the result
    return [(item - min_value) / range_value * upper_range for item in data_list]

def scale_exclude_outlier(data_list, upper_range=None):
    """
    Scale data while excluding extreme low outliers defined by the minimum thousandth percentile.

    The values below or equal to the threshold (i.e., the minimum thousandth percentile)
    are left unchanged, while the remaining data points are scaled to [0, upper_range]
    (or normalized to [0, 1] if upper_range is None).

    Parameters:
        data_list (list): List of numerical values.
        upper_range (float or None): The upper bound of the scaling range. If None, scales to [0, 1].

    Returns:
        list: The processed list with some values scaled and outliers unchanged.
    """
    # Check if data_list is a non-empty list
    if not isinstance(data_list, list) or len(data_list) == 0:
        raise ValueError("data_list must be a non-empty list")
    # If upper_range is provided, check if it is a number
    if upper_range is not None and not isinstance(upper_range, (int, float)):
        raise TypeError("upper_range must be an int or float when provided")
        
    # Determine the index for the minimum thousandth percentile threshold
    threshold_index = max(int(len(data_list) * 0.001) - 1, 0)
    # Get the threshold value by sorting the data list
    sorted_list = sorted(data_list)
    threshold_value = sorted_list[threshold_index]
    
    # Initialize minimum and maximum values using the threshold
    min_value, max_value = threshold_value, threshold_value
    
    # Update min and max values excluding the lower extreme outliers
    for item in data_list:
        if item > threshold_value:
            if item < min_value or min_value == threshold_value:
                min_value = item
            if item > max_value:
                max_value = item
                
    # Calculate the range and ensure it is not zero to prevent division errors
    range_value = max_value - min_value
    if range_value == 0:
        range_value = 1
    
    # Scale the data while preserving the order and leaving outliers unchanged
    scaled_data = []
    for item in data_list:
        if item <= threshold_value:
            # Preserve the original value for extreme low outliers
            scaled_data.append(item)
        else:
            # Scale the value to the defined range
            if upper_range is not None:
                scaled_data.append((item - min_value) / range_value * upper_range)
            else:
                scaled_data.append((item - min_value) / range_value)
    
    return scaled_data

def scale_row(row):
    """
    Scale the data in a row by its maximum value.
    If the maximum value is zero, returns the row unchanged to avoid division by zero.
    
    Parameters:
        row (array-like): A row of numerical values.
        
    Returns:
        array-like: The scaled row.
    """
    max_value = np.max(row)  # Get the maximum value in the row
    if max_value == 0:
        return row  # Return the row directly to avoid division by zero
    return row / max_value  # Return the scaled row