import numpy as np
import pandas as pd


name = 'train_FD001.txt'
df = pd.read_csv(name, sep=' ', header=None)
df.drop(columns=[26, 27], inplace=True)

var_list = df.var()
index_remove = []
for i in range(len(var_list)):
    if var_list[i] < 0.001:
        index_remove.append(i)
df.drop(columns=index_remove, inplace=True)
num_samples = len(df)

unit_list = df[0].tolist()
index_unit_change = []
for i in range(num_samples-1):
    if unit_list[i] != unit_list[i+1]:
        index_unit_change.append(i-19)       # window length is 10
index_unit_change.append(num_samples-20)
Label = np.linspace(0, 0, num_samples)
for item in index_unit_change:
    #print(item)
    for j in range(20):
        Label[item+j] = 1

df['Label'] = Label
df.drop(columns=[0, 1], inplace=True)

positive_feature_train_df = df.loc[df['Label'] == 1]
Positive_Feature_train = positive_feature_train_df.values
Positive_Feature_train = np.delete(Positive_Feature_train, -1, axis=1)
print(Positive_Feature_train.shape)
negative_feature_train_df = df.loc[df['Label'] == 0]
#negative_feature_train_df.drop(columns=['Label'], inplace=True)
Negative_Feature_train = negative_feature_train_df.values
Negative_Feature_train = np.delete(Negative_Feature_train, -1, axis=1)
print(Negative_Feature_train.shape)