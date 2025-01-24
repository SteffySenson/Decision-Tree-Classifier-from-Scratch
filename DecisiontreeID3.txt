import argparse
import csv
import math
import pandas as pd
import sys

def entropy(df): #to calculates entropy, handling single-class cases
    if df.empty:
        return 0
    counts = df[df.columns[-1]].value_counts()
    total = len(df)
    num_classes = len(counts)
    if num_classes <= 1:  # Fix: Handle single-class or empty subsets
        return 0.0  # Entropy is 0 if only one class or no examples
    entropy_value = 0
    for count in counts:
        p = count / total
        entropy_value -= p * math.log(p, num_classes)  # Correct base
    return entropy_value


def information_gain(df, attribute): #to calculates information gain
    original_entropy = entropy(df)
    weighted_entropy = 0
    for value in df[attribute].unique():
        subset = df[df[attribute] == value]
        weighted_entropy += (len(subset) / len(df)) * entropy(subset)
    return original_entropy - weighted_entropy

def majority_class(df): #to determines majority class
    return df[df.columns[-1]].mode()[0]

def id3(df, depth=0, attribute_names=None, branch_value=None, att_name=None, output=None): #to builds and prints decision tree
    if output is None:
        output = []

    df = df.dropna()

    if len(df) == 0:
        return

    current_entropy = entropy(df)
    formatted_entropy = "0.0" if current_entropy == 0 else f"{current_entropy:.16f}" 
    if current_entropy == 0.0:
        formatted_entropy = "0.0"


    if len(df[df.columns[-1]].unique()) == 1:  #pure node
        class_label = df[df.columns[-1]].iloc[0]
        output.append(f"{depth},{att_name}={branch_value},{formatted_entropy},{class_label}")
        return

    if attribute_names is None or len(attribute_names) == 0:  #no more attributes
        majority = majority_class(df)
        output.append(f"{depth},{att_name}={branch_value},{formatted_entropy},{majority}")
        return

    best_attribute = max(attribute_names, key=lambda attr: information_gain(df, attr))

    if depth == 0:
        output.append(f"{depth},root,{formatted_entropy},no_leaf")
    else:
        output.append(f"{depth},{att_name}={branch_value},{formatted_entropy},no_leaf")

    remaining_attributes = [attr for attr in attribute_names if attr != best_attribute]

    for value in sorted(df[best_attribute].unique()):
        if value != "?":
            subset = df[df[best_attribute] == value].drop(best_attribute, axis=1)
            id3(subset, depth + 1, remaining_attributes, value, best_attribute, output)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Your CSV data file", required=True)
    args = parser.parse_args()
    dataset_filename = args.data

    try:
        df = pd.read_csv(dataset_filename, header=None, na_filter=False)
    except FileNotFoundError:
        print(f"Error: Dataset file '{dataset_filename}' not found.")
        sys.exit(1)
    except pd.errors.ParserError:
        print(f"Error: Could not parse the CSV file '{dataset_filename}'.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

    df.columns = [f"att{i}" for i in range(len(df.columns) - 1)] + ['class']

    tree_output = []
    id3(df, attribute_names=df.columns[:-1], output=tree_output)

    for row in tree_output:
        print(row)

if __name__ == "__main__":
    main()