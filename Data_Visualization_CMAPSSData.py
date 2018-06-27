import numpy as np
import xgboost as xgb
from pomegranate import BayesianNetwork
import pandas as pd
from matplotlib import colors as mcolors
import seaborn as sns


colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)       # '#539caf' is a good color !

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
        index_unit_change.append(i-9)       # window length is 10
index_unit_change.append(num_samples-10)
Label = np.linspace(0, 0, num_samples)
for item in index_unit_change:
    print(item)
    for j in range(10):
        Label[item+j] = 2
        Label[item-j-1] = 1
df['Label'] = Label
df.drop(columns=[0, 1], inplace=True)

attribute_used = list(df)
print(attribute_used)
attribute_used.remove('Label')
print(attribute_used)
sp = sns.pairplot(df, hue='Label', size=3, palette=['#539caf' , colors["darkmagenta"], colors["crimson"]], vars=attribute_used)
sp.savefig(name.split('.')[0] + '_Compare.png')
