import pandas as pd
import numpy as np
import math

def entropy(class_counts):
    """Calculate entropy using base 2 (corrected)."""
    if class_counts.empty or len(class_counts) <= 1:
        return 0.0
    total = class_counts.sum()
    probs = class_counts / total
    return -(probs * np.log2(probs.where(probs > 0, 1))).sum()  # Use numpy for efficiency and handle log(0)


# Load the dataset (replace 'car.csv' with the actual path if needed)
df = pd.read_csv('car.csv', header=None, dtype=str)
df.columns = [f"att{i}" for i in range(len(df.columns) - 1)] + ['class']

# Calculate and print root entropy
root_entropy = entropy(df['class'].value_counts())
print(f"Root Entropy: {root_entropy:.16f}")