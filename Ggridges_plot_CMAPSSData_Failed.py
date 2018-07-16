import pandas as pd
import joypy
import numpy as np
from matplotlib import colors as mcolors

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

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

for j in range(len(df.columns)):
    i = 0
    for item in index_unit_change:
        if i == 0:
            data_to_add = df.iloc[item-Min_cycle+1:item+1,[j]]
            data_to_add = data_to_add.reset_index()
            data_to_add.drop(columns=['index'], inplace=True)
        else:
            append_data = df.iloc[item-Min_cycle+1:item+1,[j]]
            append_data = append_data.reset_index()
            append_data.drop(columns=['index'], inplace=True)
            data_to_add = pd.concat([data_to_add, append_data], axis=1, ignore_index=True)
        i+=1
        if i > 3:
            break
    Label = np.linspace(j, j, Min_cycle)
    data_to_add['Label'] = Label

    if j == 0:
        data_to_show = data_to_add.copy()
    else:
        data_to_show = pd.concat([data_to_show, data_to_add], axis=0, ignore_index=True)

print(data_to_show)
#fig, axes = joypy.joyplot(data_to_show, by='Label')


x_range = list(range(Min_cycle))
fig, axes = joypy.joyplot(data_to_show, by='Label', kind="values", x_range=x_range,
                       fill=False, overlap=-0.2, figsize=(8,8), grid='y')

fig_file = dir_path + '/' + 'No_Filter.png'
fig.savefig(fig_file)
