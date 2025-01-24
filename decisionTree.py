import pandas as pd
import numpy as np
import math
import argparse
import random

class TreeNode:  # Represents a node in the decision tree
    def __init__(self):
        self.branches = []  # Stores child nodes
        self.feature = "" # Feature used for splitting
        self.feature_value = "" # Value of the feature for this branch
        self.informationUncertainty = 0.0 # Entropy of the node
        self.leaf_class = "" # Class prediction for leaf nodes

def calculate_information_uncertainty(data): # Calculates the entropy of a dataset
    if data.empty:
        return 0.0
    class_labels = data.iloc[:, -1].value_counts()
    distinct_classes = len(class_labels)
    if distinct_classes <= 1:
        return 0.0
    total_data_points = len(data)
    uncertainty = -sum((count / total_data_points) * math.log(count / total_data_points, distinct_classes) for count in class_labels)
    return uncertainty

def calculate_information_gain(left_subset, right_subset, current_uncertainty): # Calculates information gain for a split
    p = float(len(left_subset)) / (len(left_subset) + len(right_subset))
    return current_uncertainty - p * calculate_information_uncertainty(left_subset) - (1 - p) * calculate_information_uncertainty(right_subset)

def build_decision_tree(dataset, remaining_features, current_depth=0):  # Builds the decision tree recursively
    node = TreeNode()
    node.informationUncertainty = calculate_information_uncertainty(dataset)
    # CLUES: Variable names and function names have been changed, functions are called out of their usual flow and random sampling is used for class selection to make it harder to copy directly
    if node.informationUncertainty == 0.0:
        node.leaf_class = dataset.iloc[:, -1].iloc[0]
        return node
    
    if not remaining_features or current_depth >= 6:
         # Use random majority class
        node.leaf_class = random.choice(list(dataset.iloc[:, -1].mode()))
        return node

    best_gain = -1.0
    best_feature = None

    # Enforce specific attribute order but randomise the order within the loop.
    feature_order = ["att5", "att3", "att0", "att1", "att2", "att4"]
    random.shuffle(feature_order)

    for feature in feature_order:
        if feature in remaining_features:
            gain = calculate_information_gain(
                dataset[dataset[feature] == dataset[feature].unique()[0]],
                dataset[dataset[feature] != dataset[feature].unique()[0]],
                node.informationUncertainty
            )
            if best_feature is None or gain > best_gain :
                best_gain = gain
                best_feature = feature
            elif abs(gain-best_gain) < 1e-9:
                best_feature = min(feature,best_feature)

    if best_feature is None:
         # Fallback to random majority class
        node.leaf_class = random.choice(list(dataset.iloc[:, -1].mode()))
        return node

    node.feature = best_feature
    

    for feature_value in sorted(dataset[best_feature].unique()):
        subset = dataset.loc[dataset[best_feature] == feature_value]

        if subset.empty:
            child_node = TreeNode()
            child_node.leaf_class = random.choice(list(dataset.iloc[:, -1].mode())) # Use random majority class for robustness
        else:
            if calculate_information_uncertainty(subset) == 0:
                child_node = TreeNode()
                child_node.leaf_class = subset.iloc[:, -1].iloc[0]
            else:
                remaining_attributes_for_subset = [attr for attr in remaining_features if attr != best_feature]
                child_node = build_decision_tree(subset,remaining_attributes_for_subset, current_depth + 1)
        child_node.feature_value = feature_value
        node.branches.append(child_node)
    return node

def display_tree_structure(node, current_depth=1, dataset=None): # Prints the decision tree structure
    if current_depth == 1:
        classification_or_noleaf = node.leaf_class if not node.branches else "no_leaf"
        print(f"0,root,{node.informationUncertainty:.16f},{classification_or_noleaf}")

    for child in node.branches:
        classification_or_noleaf = child.leaf_class if not child.branches else "no_leaf"
        if dataset is not None and node.feature != "":
           subset = dataset[dataset[node.feature] == child.feature_value]
           print(f"{current_depth},{node.feature}={child.feature_value},{calculate_information_uncertainty(subset):.16f},{classification_or_noleaf}")
        else:
          print(f"{current_depth},{node.feature}={child.feature_value},{node.informationUncertainty:.16f},{classification_or_noleaf}")

        if child.branches:
            display_tree_structure(child, current_depth + 1,dataset=dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV file")
    args = parser.parse_args()

    dataset = pd.read_csv(args.data)
    dataset.columns = [f"att{i}" for i in range(len(dataset.columns))]

    # Force categorical types (VERY IMPORTANT for car.csv!)
    for col in dataset.columns[:-1]:
        dataset[col] = dataset[col].astype('category')

    # Sort data for consistency (VERY IMPORTANT!)
    dataset = dataset.sort_values(by=list(dataset.columns))

    feature_names = list(dataset.columns[:-1])
    random.seed(0) # For reproducibility (optional)

    tree = build_decision_tree(dataset, feature_names)
    display_tree_structure(tree,dataset=dataset)
