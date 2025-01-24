import argparse
import pandas as pd
import numpy as np
import sys

def entropy(df):
    """Calculate entropy using the number of unique classes as the base for the logarithm."""
    class_counts = df[df.columns[-1]].value_counts(normalize=True)
    C = len(class_counts)  # Total number of unique classes
    if C <= 1:
        return 0.0  # If there's only one class, entropy is zero
    return -sum(p * np.log(p) / np.log(C) for p in class_counts if p > 0)

def information_gain(df, attribute, entropy_cache):
    """Calculate information gain for an attribute."""
    total_entropy = entropy_cache.get('total_entropy', entropy(df))
    entropy_cache['total_entropy'] = total_entropy  # Cache total entropy
    if attribute not in entropy_cache:
        subsets = df.groupby(attribute)
        weighted_entropy = sum((len(subset) / len(df)) * entropy(subset) for _, subset in subsets)
        entropy_cache[attribute] = weighted_entropy  # Cache weighted entropy
    return total_entropy - entropy_cache[attribute]

def majority_class(df):
    """Get majority class."""
    return df[df.columns[-1]].mode()[0]

def id3(df, depth=0, attributes=None, parent_split=None, output=None, entropy_cache=None):
    """Build decision tree."""
    if output is None:
        output = []
    if entropy_cache is None:
        entropy_cache = {}

    # Check for pure node
    if len(df[df.columns[-1]].unique()) == 1:
        class_label = df.iloc[0, -1]
        output.append(f"{depth},{parent_split},{0.0},{class_label}")
        return

    # Check for no attributes left
    if not attributes:
        majority = majority_class(df)
        output.append(f"{depth},{parent_split or 'root'},{entropy(df):.16f},{majority}")
        return

    # Find best attribute
    best_attribute = max(attributes, key=lambda attr: information_gain(df, attr, entropy_cache))
    output.append(f"{depth},{parent_split or 'root'},{entropy(df):.16f},no_leaf")

    # Recursively split data
    for value, subset in df.groupby(best_attribute):
        id3(subset.drop(columns=best_attribute), depth + 1, 
            [attr for attr in attributes if attr != best_attribute], 
            f"{best_attribute}={value}", output, entropy_cache)

def preprocess_data(df):
    """Preprocess the data: handle missing values and convert to appropriate types."""
    # Fill missing values with the mode of each column
    for column in df.columns:
        mode_value = df[column].mode()[0]
        df[column] = df[column].fillna(mode_value)

    # Convert categorical columns to string type
    for column in df.columns:
        df[column] = df[column].astype(str)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=True, help="Path to the dataset (CSV format)")
    parser.add_argument("-o", "--output", help="Path to save the output (optional)")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.data, header=None, dtype=str)
    except FileNotFoundError:
        print(f"Error: File '{args.data}' not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        sys.exit(1)
    except pd.errors.ParserError:
        print("Error: The file could not be parsed. Please check the format.")
        sys.exit(1)

    # Check for at least two columns (features + class)
    if df.shape[1] < 2:
        print("Error: The dataset must contain at least one feature and one class column.")
        sys.exit(1)

    # Set column names
    df.columns = [f"att{i}" for i in range(len(df.columns) - 1)] + ['class']

    # Preprocess the data
    preprocess_data(df)

    # Run ID3 algorithm
    output = []
    id3(df, attributes=df.columns[:-1].tolist(), output=output)

    # Save or print the output
    if args.output:
        with open(args.output, 'w') as f:
            f.write("\n".join(output))
    else:
        print("\n".join(output))

if __name__ == "__main__":
    main()