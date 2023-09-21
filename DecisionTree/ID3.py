import Tree
import pandas as pd
import math
import numpy as np

def ID3(S, Attributes, Label):
    all_same_label = True
    labels = []
    for example in S:
        if example[-1] not in labels:
            labels.append(example[-1])
        if len(labels) > 1:
            all_same_label = False
            break

    if all_same_label:
        return Tree.Node(labels[0])

def readData(file):
    with open(file, 'r') as f:
        for line in f:
            terms = line.strip().split(',')


def entropy(data, vals):
    value_counts = {v: 0 for v in vals}

    for example in data:
        label_val = example[-1]
        value_counts[label_val] += 1
    entropy = 0
    for v in value_counts:
        p_i = value_counts[v] / len(data)
        entropy += p_i * math.log(p_i, 2)
    return -entropy


def informationGain(data, attr, vals):
    gain = entropy(data, vals)
    attr_values = np.unique(data[attr])
    for av in attr_values:


def getLabelValues(data):
    values = []
    for example in data:
        if example[-1] not in values:
            values.append(example[-1])