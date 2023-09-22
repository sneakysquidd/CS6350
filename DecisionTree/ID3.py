import Tree
import pandas as pd
import math
import numpy as np


def ID3(S, Attributes, func, depth, max_depth):
    labels = np.unique(S.iloc[:, -1:])
    if len(labels) == 1:
        return Tree.Node(labels[0], depth + 1)
    if len(Attributes) == 0:
        return Tree.Node(S.iloc[:, -1:].mode().iloc[0,0], depth + 1)

    best_gain = 0
    best_a = -1
    for a in Attributes:
        ig = informationGain(S, a)
        if ig > best_gain:
            best_gain = ig
            best_a = a

    root = Tree.Node(best_a, depth)
    attr_vals = np.unique(S[best_a])

    # for each possible value that the best attribute can take
    for av in attr_vals:
        branch = Tree.Node(av, depth)
        av_data = S[S[best_a] == av]
        if depth == max_depth - 1:
            if av_data.empty:
                branch.children.append(Tree.Node(S.iloc[:, -1:].mode().iloc[0, 0]), depth + 1)
            else:
                branch.children.append(Tree.Node(av_data.iloc[:, -1:].mode().iloc[0, 0]), depth + 1)
        else:
            if av_data.empty:
                branch.children.append(Tree.Node(S.iloc[:, -1:].mode().iloc[0, 0]), depth + 1)
            else:
                unused_attributes = Attributes.copy()
                unused_attributes.remove(best_a)
                branch.children.append(ID3(av_data, unused_attributes, depth + 1))
        root.children.append(branch)
    return root


def entropy(data):
    vals = np.unique(data.iloc[:, -1:])

    val_counts = data.iloc[:, -1:].value_counts()
    value_counts = {i[0][0]: i[1] for i in val_counts.items()}

    e = 0
    for v in value_counts:
        p_i = value_counts[v] / len(data)
        e += p_i * math.log(p_i, 2)
    return -e


def majorityError(data):
    total = data.shape[0]
    val_counts = data.iloc[:, -1:].value_counts()
    value_counts = [i[1] for i in val_counts.items()]
    return (total - max(value_counts)) / total


def giniIndex(data):
    total = data.shape[0]
    val_counts = data.iloc[:, -1:].value_counts()
    value_props = [i[1] / total for i in val_counts.items()]
    gi = 1
    for p in value_props:
        gi -= p**2


def informationGain(data, attr, func):
    purity_func = lambda x : x
    if func == 0:
        purity_func = entropy
    if func == 1:
        purity_func = majorityError
    if func == 2:
        purity_func = giniIndex

    ig = purity_func(data)
    attr_values = np.unique(data[attr])
    for av in attr_values:
        attr_data = data[data[attr] == av]
        attr_purity = purity_func(attr_data)
        ig -= (len(attr_data) / len(data)) * attr_purity
    return ig


data2 = pd.read_csv("E:\\2023 school\\6350\decisionTree\car-4\\test.csv", header=None)
attributes = [i for i in range(len(data2.columns) - 1)]
ID3(data2, attributes)