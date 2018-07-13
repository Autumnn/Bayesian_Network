import pandas as pd
import numpy as np

name = 'train_FD001.txt'
print(name)

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
max_cycle = []
for i in range(num_samples-1):
    if unit_list[i] != unit_list[i+1]:
        index_unit_change.append(i)       # window length is 10
        max_cycle.append(df.ix[i,1])
#        print(df.ix[i,1])
Min_cycle = min(max_cycle)
index_unit_change.append(num_samples-1)
Label = np.linspace(0, 0, num_samples)
for item in index_unit_change:
#    print(item)
    for j in range(Min_cycle):
#        Label[item+j] = 2
        Label[item-j] = 1
df['Label'] = Label
print(df)

df.drop(columns=[0, 1], inplace=True)
