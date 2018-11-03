import pandas as pd
import numpy as np
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
    i= 0
    for item in index_unit_change:
        data_to_add = df.iloc[item - Min_cycle + 1:item + 1, [j]]
        data_to_add = data_to_add.reset_index()
        data_to_add.drop(columns=['index'], inplace=True)
        sz = (Min_cycle,)  # size of array
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
        xhat[0] = data_to_add.iloc[0, [0]]
        for k in range(1, Min_cycle):
            # time update
            xhatminus[k] = xhat[k - 1]
            Pminus[k] = P[k - 1] + Q
            # measurement update
            K[k] = Pminus[k] / (Pminus[k] + R)
            xhat[k] = xhatminus[k] + K[k] * (data_to_add.iloc[k, [0]] - xhatminus[k])
            P[k] = (1 - K[k]) * Pminus[k]
        if i ==0:
            data_to_show = pd.DataFrame(xhat)
            i += 1
        else:
            data_to_show = pd.concat([data_to_show, pd.DataFrame(xhat)], axis=1, ignore_index=True)
    data_cut = data_to_show.iloc[20:]
    print(data_cut)
    if j in [0,1,2,4,5,6,8,9,10,11]:
        Min_value = data_cut.min()
        for k in range(len(data_cut.columns)):
            ax.plot(data_cut.iloc[:, [k]] - Min_value[k])
    else:
        Max_value = data_cut.max()
        for k in range(len(data_cut.columns)):
            ax.plot(data_cut.iloc[:, [k]] - Max_value[k])


fig_file = dir_path + '/' + 'Kalman_Filter_shift.png'
fig.savefig(fig_file)
