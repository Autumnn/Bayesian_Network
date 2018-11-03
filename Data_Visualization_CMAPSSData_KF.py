import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
import seaborn as sns


colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)       # '#539caf' is a good color !

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
index_unit_change.append(num_samples-1)
max_cycle.append(df.ix[num_samples-1,1])
df.drop(columns=[0, 1], inplace=True)
num = len(index_unit_change)

initial = True
for j in range(len(df.columns)):
    column_ini = True
    for item in index_unit_change:
        idx = index_unit_change.index(item)
        data_to_filter = df.iloc[item - max_cycle[idx] + 1:item + 1, [j]]
        data_to_filter = data_to_filter.reset_index()
        data_to_filter.drop(columns=['index'], inplace=True)
        sz = (max_cycle[idx],)  # size of array
        Q = 1e-5  # process variance
        # allocate space for arrays
        xhat = np.zeros(sz)  # a posteri estimate of x
        P = np.zeros(sz)  # a posteri error estimate
        xhatminus = np.zeros(sz)  # a priori estimate of x
        Pminus = np.zeros(sz)  # a priori error estimate
        K = np.zeros(sz)  # gain or blending factor
        R = 0.1 ** 2  # estimate of measurement variance, change to see effect
        # intial guesses
        P[0] = 1.0
        xhat[0] = data_to_filter.iloc[0, [0]]
        for k in range(1, max_cycle[idx]):
            # time update
            xhatminus[k] = xhat[k - 1]
            Pminus[k] = P[k - 1] + Q
            # measurement update
            K[k] = Pminus[k] / (Pminus[k] + R)
            xhat[k] = xhatminus[k] + K[k] * (data_to_filter.iloc[k, [0]] - xhatminus[k])
            P[k] = (1 - K[k]) * Pminus[k]
        if column_ini:
            column_ini = False
            new_col = pd.DataFrame(xhat)
        else:
            new_col = pd.concat([new_col, pd.DataFrame(xhat)],axis=0,ignore_index=True)
    if initial:
        initial = False
        new_df = new_col.copy()
    else:
        new_df = pd.concat([new_df, new_col], axis=1, ignore_index=True)

Label = np.linspace(0, 0, num_samples)
for item in index_unit_change:
    print(item)
    for j in range(10):
        Label[item-j] = 1
new_df['Label'] = Label

index_unit_change.pop()
index_unit_change.insert(0, -1)
for item in index_unit_change:
    idx = index_unit_change.index(item)
    new_df.drop(new_df.index[item+1-idx*20: item+21-idx*20], axis=0, inplace=True)      # delete the first 20 cycles, which are the initial stage of Kalman Filter

attribute_used = list(new_df)
print(attribute_used)
attribute_used.remove('Label')
print(attribute_used)
sp = sns.pairplot(new_df, hue='Label', size=3, palette=['#539caf' , colors["darkmagenta"], colors["crimson"]], vars=attribute_used)
sp.savefig(name.split('.')[0] + '_Compare_KF.png')