import Tree
import pandas as pd
import math
import numpy as np


def ID3(S, Attributes):
    labels = np.unique(S.iloc[:, -1:])
    if len(labels) == 1:
        return Tree.Node(labels[0])
    if len(Attributes) == 0:
        return Tree.Node(S.iloc[:, -1:].mode().iloc[0,0])

    best_gain = 0
    best_a = -1
    for a in Attributes:
        ig = informationGain(S, a)
        if ig > best_gain:
            best_gain = ig
            best_a = a

    root = Tree.Node(best_a)
    attr_vals = np.unique(S[best_a])

    # for each possible value that the best attribute can take
    for av in attr_vals:
        branch = Tree.Node(av)
        av_data = S[S[best_a] == av]
        if av_data.empty:
            branch.children.append(Tree.Node(S.iloc[:, -1:].mode().iloc[0, 0]))
        else:
            unused_attributes = Attributes.copy()
            unused_attributes.remove(best_a)
            branch.children.append(ID3(av_data, unused_attributes))
        root.children.append(branch)
    return root

def readData(file):
    with open(file, 'r') as f:
        for line in f:
            terms = line.strip().split(',')


def entropy(data):
    vals = np.unique(data.iloc[:, -1:])

    val_counts = data.iloc[:, -1:].value_counts()
    value_counts = {i[0][0]: i[1] for i in val_counts.items()}

    e = 0
    for v in value_counts:
        p_i = value_counts[v] / len(data)
        e += p_i * math.log(p_i, 2)
    return -e


def informationGain(data, attr):
    ig = entropy(data)
    attr_values = np.unique(data[attr])
    for av in attr_values:
        attr_data = data[data[attr] == av]
        attr_entropy = entropy(attr_data)
        ig -= (len(attr_data) / len(data)) * attr_entropy
    return ig

data = pd.read_csv("E:\\2023 school\\6350\decisionTree\car-4\\test.csv", header=None)
attributes = [i for i in range(len(data.columns) - 1)]
ID3(data, attributes)