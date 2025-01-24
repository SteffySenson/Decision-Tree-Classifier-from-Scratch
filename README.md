# Decision Tree Classifier from Scratch

This repository contains a Python implementation of a decision tree classifier using the ID3 algorithm.  The code focuses on clarity and understanding of the algorithm's inner workings rather than relying on pre-built machine learning libraries. 

## Introduction

Decision trees are a fundamental tool in machine learning, used for both classification and regression tasks.  This project provides a hands-on approach to learning how these trees are constructed, using the information gain criterion to select the best attributes for splitting data. 

This specific implementation is designed to handle discrete (categorical) features and will provide you with a good understanding of how the ID3 algorithm works. 

## Key Features

*   **ID3 Algorithm:** Implements the core logic of the ID3 decision tree learning algorithm.
*   **Entropy Calculation:** Uses the standard entropy formula to measure impurity of a data set.
*   **Information Gain:** Calculates information gain to choose the best attribute for splitting at each node.
*   **Discrete Features:** Designed to work with datasets where features take discrete (categorical) values.
*   **No External Libraries:** Avoids reliance on higher-level machine learning libraries for a more fundamental learning experience (you may use `numpy` and `ElementTree` if desired).
*   **Clean Output:** The tree is output with each node on its own line, making it easy to understand the tree structure. The format is as follows: depth, attname=value, entropy, class.

## Usage

1.  **Clone the repository:**

    ```bash
    git clone [repository_url]
    cd [repository_name]
    ```

2.  **Prepare your data:**

    Your input data should be in CSV format where **the last column of each row is the class label and each attribute should be a separate column**. The program expects the data to be in this format:

    ```csv
    feature1_value1, feature2_value1, ..., class1
    feature1_value2, feature2_value2, ..., class2
    ...
    ```

3.  **Run the program:**

    ```bash
    python3 student.py --data <path_to_your_data.csv>
    ```
    
    For example, to run the program on a dataset called `my_data.csv`:
    
     ```bash
     python3 student.py --data my_data.csv
     ```
    
    The `--data` parameter specifies the location of your CSV file.

## Output Format

The program will output the decision tree structure to standard output. Each line represents a node in the tree:

*   **depth**: The depth of the node in the tree. The root node has a depth of 0.
*   **attname=value**: The name of the attribute being split on at that node (e.g. att1) and the corresponding value (e.g. value1).
*   **entropy**: The entropy of the data at that node. 
*   **class**: The predicted class or `no_leaf` if the node has children.

The root node will have depth 0 and have `attname` as an empty string.

For example:
    ```
    0, , 1.10, no_leaf
    1,att0=value1,0.85, no_leaf
    2,att1=value2, 0.0, class1
    1, att0=value2,0.0, class2
    ```
    This implies that at the root node with depth 0, the entropy is 1.10. When you split the data on attribute 0 with value "value1", you go to the node that has depth 1 and has an entropy of 0.85. Then you split the data on attribute 1 with value "value2" at depth 2 and that leads to class "class1". At depth 1, if you had split on attribute 0 with value "value2" you would reach "class2".

## Understanding the Code

The code is structured to clearly demonstrate the following stages of the ID3 algorithm:

*   **Data Loading:** Reading the provided CSV file.
*   **Entropy Calculation:** Implementing the formula to calculate the entropy of a data set.
*   **Attribute Selection:** Iterating through possible splitting attributes.
*   **Information Gain Calculation:** Computing information gain for each attribute to select the best one for a split.
*   **Tree Construction:** Building the decision tree recursively.
*   **Output Tree Structure:** Formatting the tree to standard output.

## Further Exploration

*   **Handling Missing Values:** Extend the code to deal with missing values in the dataset.
*   **Pruning:** Implement pruning techniques to avoid overfitting of the decision tree.
*   **Visualizing Trees:** Use a graphing library to visually represent the tree.
*   **Different Datasets:** Try the program on different datasets.

## Contribution

Feel free to contribute to this project by reporting issues or submitting pull requests.

## License

MIT License
