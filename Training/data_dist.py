import pandas as pd

df = pd.read_csv("Training/data_indices/train_indice.csv")

print(df.shape)

print(df.label.value_counts())

df = pd.read_csv("Training/data_indices/test_indice.csv")

print(df.shape)

print(df.label.value_counts())