import numpy as np
import os

# These stay as safety guards
os.environ['OPENBLAS_NUM_THREADS'] = '1'

print("Testing Matrix Math...")
A = np.random.rand(10, 10)
B = np.random.rand(10, 1)

try:
    # This is the exact operation that was causing the crash
    C = A @ B
    print("✅ Math successful! No Segmentation Fault.")
except Exception as e:
    print(f"❌ Error: {e}")