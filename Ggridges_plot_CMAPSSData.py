import pandas as pd
from matplotlib import pyplot as plt


name = 'train_FD001.txt'
print(name)
dir_path = 'CMAPSSData_Plot'

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
df.drop(columns=[0, 1], inplace=True)
num = len(index_unit_change)

fig = plt.figure(figsize=(8,72))
Title = "Feature Test"
fig.canvas.set_window_title(Title)
fig.subplots_adjust(hspace=1.2)

for j in range(len(df.columns)):
    ax = plt.subplot(len(df.columns) + 1, 1, j + 1)
    for item in index_unit_change:
        data_to_add = df.iloc[item - Min_cycle + 1:item + 1, [j]]
        data_to_add = data_to_add.reset_index()
        data_to_add.drop(columns=['index'], inplace=True)
        ax.plot(data_to_add)

fig_file = dir_path + '/' + 'No_Filter.png'
fig.savefig(fig_file)
