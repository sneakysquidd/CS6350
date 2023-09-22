import pandas as pd
import numpy as np

b = pd.read_csv("E:\\2023 school\\6350\decisionTree\car-4\\train.csv", header=None, names=["buying","maint","doors","persons","lug_boot","safety","label"])
print(np.unique(b["label"])[0])
print(b["label"].mode()[0])

