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

    best_attribute = max(attributes, key=lambda attr: information_gain(data, attr, class_attr, purity_func))
    root = Node(attribute=best_attribute)

    if data.dtypes[best_attribute] == np.int64 or data.dtypes[best_attribute] == np.float64:
        median = data[best_attribute].median()
        data[best_attribute] = pd.cut(data[best_attribute], [-float("inf"), median, float("inf")], labels=[f"<= {median}", f"> {median}"])

    for value in np.unique(data[best_attribute]):
        subset = data.where(data[best_attribute] == value).dropna()
        if len(subset) == 0:
            root.children[value] = Node(label=data[class_attr].mode().iloc[0])
        else:
            root.children[value] = id3(subset, [attr for attr in attributes if attr != best_attribute], class_attr, purity_func, max_depth, curr_depth + 1)

    return root


def predict(tree, data_point):
    if tree.label is not None:
        return tree.label
    attribute = tree.attribute

    if data_point.dtypes[attribute] == np.int64 or data_point.dtypes[attribute] == np.float64:
        for attribute_val in tree.children:
            if eval(data_point[attribute] + attribute_val):
                data_value = attribute_val
    else:
        data_value = data_point[attribute]
    if data_value not in tree.children:
        child_labels = [child.label for child in tree.children.values()]
        return max(set(child_labels), key=child_labels.count)
    return predict(tree.children[data_point[attribute]], data_point)


train_data = pd.read_csv("E:\\2023 school\\6350\decisionTree\\bank-4\\train.csv", header=None, names=["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome","y"])
test_data = pd.read_csv("E:\\2023 school\\6350\decisionTree\\bank-4\\test.csv", header=None, names=["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome","y"])
tree = id3(train_data, ["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome"], "y", entropy, max_depth = 2)


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

