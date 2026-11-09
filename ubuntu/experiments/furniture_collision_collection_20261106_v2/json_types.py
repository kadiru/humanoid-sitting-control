import numpy as np

def encode(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    raise TypeError(f'Unsupported JSON type: {type(value).__name__}')
