import Tree
import pandas as pd
import math
import numpy as np


def ID3(S, Attributes, label, func, depth, max_depth):
    labels = np.unique(S[label])
    if len(labels) == 1:
        temp = Tree.Node(labels[0], depth + 1)
        temp.isLeaf = True
        return temp
    if len(Attributes) == 0:
        temp = Tree.Node(S[label].mode()[0], depth + 1)
        temp.isLeaf = True
        return temp

    best_gain = 0
    best_a = ""
    for a in Attributes:
        ig = informationGain(S, a, func, label)
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
                temp = Tree.Node(S[label].mode()[0], depth + 1)
                temp.isLeaf = True
                branch.children.append(temp)
            else:
                temp = Tree.Node(av_data[label].mode()[0], depth + 1)
                temp.isLeaf = True
                branch.children.append(temp)
        else:
            if av_data.empty:
                temp = Tree.Node(S[label].mode()[0], depth + 1)
                temp.isLeaf = True
                branch.children.append(temp)
            else:
                unused_attributes = Attributes.copy()
                unused_attributes.remove(best_a)
                branch.children.append(ID3(av_data, unused_attributes, label, func, depth + 1, max_depth))
        root.children.append(branch)
    return root


def entropy(data, label):
    vals = np.unique(data.iloc[:, -1:])

    val_counts = data[label].value_counts()
    value_counts = {i[0][0]: i[1] for i in val_counts.items()}

    e = 0
    for v in value_counts:
        p_i = value_counts[v] / len(data)
        e += p_i * math.log(p_i, 2)
    return -e


def majorityError(data, label):
    total = data.shape[0]
    val_counts = data[label].value_counts()
    value_counts = [i[1] for i in val_counts.items()]
    return (total - max(value_counts)) / total


def giniIndex(data, label):
    total = data.shape[0]
    val_counts = data[label].value_counts()
    value_props = [i[1] / total for i in val_counts.items()]
    gi = 1
    for p in value_props:
        gi -= p**2
    return gi


def informationGain(data, attr, func, label):
    purity_func = lambda x : x
    if func == 0:
        purity_func = entropy
    if func == 1:
        purity_func = majorityError
    if func == 2:
        purity_func = giniIndex

    ig = purity_func(data, label)
    attr_values = np.unique(data[attr])
    for av in attr_values:
        attr_data = data[data[attr] == av]
        attr_purity = purity_func(attr_data, label)
        ig -= (len(attr_data) / len(data)) * attr_purity
    return ig


def classify(root, data):
    for child in root.children:
        if child.data == data[0]:
            


test_data = pd.read_csv("E:\\2023 school\\6350\decisionTree\car-4\\test.csv", header=None, names=["buying","maint","doors","persons","lug_boot","safety","label"])
train_data = pd.read_csv("E:\\2023 school\\6350\decisionTree\car-4\\train.csv", header=None, names=["buying","maint","doors","persons","lug_boot","safety","label"])
attributes = ["buying","maint","doors","persons","lug_boot","safety"]
for i in range(1,7):
    tree = ID3(train_data, attributes, "label", 0, 0, i)
    for index, r in test_data.iterrows():
        #

