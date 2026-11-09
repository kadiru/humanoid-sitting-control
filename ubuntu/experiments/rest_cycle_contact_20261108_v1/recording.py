"""Lossless JSON-compatible values; never stringify a rejected audit result."""
import json
import numpy as np

def encode(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    raise TypeError(type(value).__name__)

def write(path,value):
    payload=json.dumps(value,indent=2,default=encode,allow_nan=False)+'\n'
    with path.open('x') as f:f.write(payload)

