import numpy as np

def convert_to_binary(y):
    """
    Converts ECG labels to binary classification.
    
    Mapping:
    0: Normal (N)
    1: Abnormal (S, V, Q, F, etc.)
    
    Correctly handles both raw strings ('N', 'S', ...) and potential 
    intermediate numeric encodings where 'N' has been mapped to 0.
    """
    # If already numeric and 'Normal' is 0
    if np.issubdtype(y.dtype, np.number):
        return np.where(y == 0, 0, 1)
    
    # Otherwise assume strings from MIT-BIH
    return np.where(y == 'N', 0, 1)
