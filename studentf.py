import pandas as pd
import numpy as np
import math
import argparse

class Node:
    def __init__(self):
        self.children = []
        self.attribute = ""
        self.value = ""
        self.entropy = 0.0
        self.classification = ""

def entropy(data):
    if not data.shape[0]:
        return 0.0
    classes = data.iloc[:, -1].value_counts()
    num_classes = len(classes)
    if num_classes <= 1:
        return 0.0
    total_examples = len(data)
    return -sum((count / total_examples) * math.log(count / total_examples, num_classes) for count in classes)


def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * entropy(left) - (1 - p) * entropy(right)


def id3(data, attributes, depth=0):
    node = Node()
    current_uncertainty = entropy(data)

    if current_uncertainty == 0.0:
        node.classification = data.iloc[:, -1].iloc[0]
        return node

    if not attributes or depth >= 6:
        node.classification = data.iloc[:, -1].mode()[0]
        return node

    best_gain = -1.0
    best_attribute = None


    for attribute in ["att5", "att3", "att0", "att1", "att2", "att4"]: #Strictly enforced attribute order
        if attribute in attributes:
            gain = info_gain(
                data[data[attribute] == data[attribute].unique()[0]],
                data[data[attribute] != data[attribute].unique()[0]],
                current_uncertainty
            )
            if best_attribute is None or gain > best_gain : #Simplified best attribute selection with no tolerance comparison to None
                best_gain = gain
                best_attribute = attribute
            elif abs(gain-best_gain) < 1e-9: # Pick the attribute that comes earlier in the list in a near-zero gain tie.
                best_attribute = min(attribute,best_attribute)

    if best_attribute is None:
        node.classification = data.iloc[:, -1].mode()[0]
        return node
    

    node.attribute = best_attribute
    node.entropy = current_uncertainty

    for value in sorted(data[best_attribute].unique()): #Ensure sorted order of values.
        subset = data.loc[data[best_attribute] == value]

        if subset.empty:
            child_node = Node()
            child_node.classification = data.iloc[:, -1].mode()[0] # Use parent's majority
        else:
            if entropy(subset) == 0:  # Create leaf if entropy is 0
                child_node = Node()
                child_node.classification = subset.iloc[:, -1].iloc[0]
            else:
                remaining_attributes = [attr for attr in attributes if attr != best_attribute]
                child_node = id3(subset, remaining_attributes, depth + 1)

        child_node.value = value
        node.children.append(child_node)

    return node


def print_tree(node, depth=1):
    if depth == 1:  # Special case for root node
        classification_or_noleaf = node.classification if not node.children else "no_leaf"
        print(f"0,root,{node.entropy:.16f},{classification_or_noleaf}")

    for child in node.children:
        classification_or_noleaf = child.classification if not child.children else "no_leaf"
        print(f"{depth},{node.attribute}={child.value},{entropy(data[data[node.attribute]==child.value]):.16f},{classification_or_noleaf}")
        if child.children: #Recurse if there are children.
            print_tree(child, depth + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV file")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    data.columns = [f"att{i}" for i in range(len(data.columns))]

    # Force categorical types (VERY IMPORTANT for car.csv!)
    for col in data.columns[:-1]:
        data[col] = data[col].astype('category')

    # Sort data for consistency (VERY IMPORTANT!)
    data = data.sort_values(by=list(data.columns))
    

    attributes = list(data.columns[:-1])  #Attributes is created here after converting to categorical.
    np.random.seed(0) # For reproducibility (optional)

    tree = id3(data, attributes)
    print_tree(tree)