import sys

import pandas as pd
import numpy as np


class Node:
    def __init__(self, attribute=None, label=None):
        self.attribute = attribute
        self.label = label
        self.children = {}


def entropy(data):
    unique, counts = np.unique(data, return_counts=True)
    probabilities = counts / len(data)
    return -np.sum(probabilities * np.log2(probabilities))


def majority_error(data):
    total = data.shape[0]
    unique, counts = np.unique(data, return_counts=True)
    return (total - max(counts)) / total


def gini_index(data):
    total = data.shape[0]
    unique, counts = np.unique(data, return_counts=True)
    value_props = counts / total
    gi = 1
    for p in value_props:
        gi -= p**2
    return gi


def information_gain(data, attribute, class_attr, purity_func):
    total_entropy = purity_func(data[class_attr])
    values, counts = np.unique(data[attribute], return_counts=True)
    weighted_entropy = np.sum(
        [(counts[i] / len(data)) * entropy(data.where(data[attribute] == values[i]).dropna()[class_attr]) for i in
         range(len(values))])
    return total_entropy - weighted_entropy


def id3(data, attributes, class_attr, purity_func, max_depth = 9999999, curr_depth = 0):
    if curr_depth >= max_depth or len(attributes) == 0:
        return Node(label=data[class_attr].mode().iloc[0])
    if len(np.unique(data[class_attr])) == 1:
        return Node(label=data[class_attr].iloc[0])

    # Find the best attribute to split on based on the passed in purity function
    best_attribute = max(attributes, key=lambda attr: information_gain(data, attr, class_attr, purity_func))
    root = Node(attribute=best_attribute)

    # If the attribute is numeric, process it based on the median
    if data.dtypes[best_attribute] == np.int64 or data.dtypes[best_attribute] == np.float64:
        median = data[best_attribute].median()
        data[best_attribute] = pd.cut(data[best_attribute], [-float("inf"), median, float("inf")], labels=[f"<= {median}", f"> {median}"])

    # Create branches for each value of best splitting attribute
    for value in np.unique(data[best_attribute]):
        subset = data.where(data[best_attribute] == value).dropna()
        if len(subset) == 0:
            root.children[value] = Node(label=data[class_attr].mode().iloc[0])
        else:
            root.children[value] = id3(subset, [attr for attr in attributes if attr != best_attribute], class_attr, purity_func, max_depth, curr_depth + 1)

    return root


def predict(tree, data_point, dtypes):
    if tree.label is not None:
        return tree.label
    attribute = tree.attribute

    if dtypes[attribute] == np.int64 or dtypes[attribute] == np.float64:
        data_value = list(tree.children.keys())[0]
        for attribute_val in tree.children:
            if eval(f"{data_point[attribute]}" + attribute_val):
                data_value = attribute_val
    else:
        data_value = data_point[attribute]
    if data_value not in tree.children:
        child_labels = [child.label for child in tree.children.values()]
        return max(set(child_labels), key=child_labels.count)
    return predict(tree.children[data_value], data_point, dtypes)

def build_tree(train_data, test_data, max_depth, func_id, columns, label):
    train = pd.read_csv(train_data, header=None, names=columns)
    test = pd.read_csv(test_data, header=None, names=columns)
    attributes = columns.copy().remove(label)
    if func_id == 0:
        purity_func = entropy
    elif func_id == 1:
        purity_func = majority_error
    else:
        purity_func = gini_index
    tree = id3(train, attributes, label, purity_func, max_depth)

    total = 0
    wrong = 0
    dtypes = train_data.dtypes

    for index, r in test_data.iterrows():
        prediction = predict(tree, r, dtypes)
        if prediction != r["y"]:
            wrong += 1
        total += 1
    return f"Error %: {wrong / total} max depth: {i} Data: Test"


def main():
    if len(sys.argv) != 7:
        print("Usage: ID3.py <train_data_path>, <test_data_path>, <max_depth>, <func_id>, <columns>, <label>")
        sys.exit(1)

    print(build_tree(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]))


if __name__ == "__main__":
    main()

#Code used to generate error percentages based on maximum tree depth

# train_data = pd.read_csv("E:\\2023 school\\6350\decisionTree\\bank-4\\train.csv", header=None, names=["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome","y"])
# test_data = pd.read_csv("E:\\2023 school\\6350\decisionTree\\bank-4\\test.csv", header=None, names=["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome","y"])
# #tree = id3(train_data, ["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome"], "y", entropy, max_depth = 2)
#
# for column in train_data:
#     mode = train_data.where(train_data[column] != "unknown")[column].mode()[0]
#     train_data.loc[train_data[column] == "unknown", column] = mode
#     test_data.loc[test_data[column] == "unknown", column] = mode
#
# total = 0
# wrong = 0
# dtypes = train_data.dtypes
#
# for i in range(1,17):
#     tree = id3(train_data.copy(), ["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome"], "y", majority_error, i)
#     for index, r in test_data.iterrows():
#         prediction = predict(tree, r, dtypes)
#         if prediction != r["y"]:
#             wrong += 1
#         total += 1
#     print(f"Error %: {wrong/total} max depth: {i} Data: Test")
#
#     for index, r in train_data.iterrows():
#         prediction = predict(tree, r, dtypes)
#         if prediction != r["y"]:
#             wrong += 1
#         total += 1
#     print(f"Error %: {wrong/total} max depth: {i} Data: Train")

# test_data = pd.read_csv("E:\\2023 school\\6350\decisionTree\car-4\\test.csv", header=None, names=["buying","maint","doors","persons","lug_boot","safety","label"])
# train_data = pd.read_csv("E:\\2023 school\\6350\decisionTree\\bank-4\\train.csv", header=None, names=["buying","maint","doors","persons","lug_boot","safety","label"])
# attributes = ["buying","maint","doors","persons","lug_boot","safety"]
# tree = id3(train_data, attributes, "label", entropy)
#
# total = 0
# wrong = 0
#
#
# for i in range(1,7):
#     tree = id3(train_data, attributes, "label", gini_index, i)
#     for index, r in test_data.iterrows():
#         prediction = predict(tree, r)
#         if prediction != r["label"]:
#             wrong += 1
#         total += 1
#     print(f"Error %: {wrong/total} max depth: {i} Data: Test")
#     for index, r in train_data.iterrows():
#         prediction = predict(tree, r)
#         if prediction != r["label"]:
#             wrong += 1
#         total += 1
#     print(f"Error %: {wrong/total} max depth: {i} Data: Train")

