import pandas as pd
import numpy as np

b = pd.read_csv("E:\\2023 school\\6350\decisionTree\car-4\\test.csv", header=None)
total = b.shape[0]
val_counts = b.iloc[:, -1:].value_counts()
value_counts = [i[1] / total for i in val_counts.items()]
print(value_counts)

