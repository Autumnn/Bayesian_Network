import numpy as np
import pandas as pd


def transfer(data, bounds, num_features, nominal_feature):
    size = data.shape[0]
    for k in range(size):
        initial_sample = data[k, :]
        z = np.linspace(0, 0, num_features)
        for ii in range(num_features):
            if ii not in nominal_feature:
                index_array = np.where(bounds[:, ii] <= initial_sample[ii])[0]
                if index_array.shape[0] == 0:
                    z[ii] = 0
                else:
                    z[ii] = np.max(index_array)
                if z[ii] > 99:
                    z[ii] -= 1
            else:
                z[ii] = initial_sample[ii]
        data[k, :] = z


threshold = 21

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
        index_unit_change.append(i-threshold-1)       # window length is threshold
index_unit_change.append(num_samples-threshold)
Label = np.linspace(0, 0, num_samples)
for item in index_unit_change:
    #print(item)
    for j in range(threshold):
        Label[item+j] = 1

df['Label'] = Label
df.drop(columns=[0, 1], inplace=True)

name_test = 'test_FD001.txt'
df_test = pd.read_csv(name_test, sep=' ', header=None)
df_test.drop(columns=[26, 27], inplace=True)
df_test.drop(columns=index_remove, inplace=True)
num_samples_test = len(df_test)
Label_RUL = np.loadtxt('RUL_FD001.txt')

unit_list = df_test[0].tolist()
dic_initialed = True
for i in range(num_samples_test-1):
    if unit_list[i] != unit_list[i+1] or i == num_samples_test-2:
        if i == num_samples_test-2:
            i += 1
        unit_index = unit_list[i]
        remain_cycles = Label_RUL[unit_index-1]
        print(unit_index, remain_cycles)
        if remain_cycles <= threshold:
            changed_label_num = threshold - remain_cycles
            if dic_initialed:
                index_test_change = {i: changed_label_num}
                dic_initialed = False
            else:
                index_test_change[i] = changed_label_num

Label_test = np.linspace(0, 0, num_samples_test)
for key, value in index_test_change.items():
    #print(key, value)
    for j in range(int(value)):
        Label_test[key-j] = 1

df_test['Label'] = Label_test
df_test.drop(columns=[0, 1], inplace=True)

positive_feature_train_df = df.loc[df['Label'] == 1]
Positive_Feature_train = positive_feature_train_df.values
Positive_Feature_train = np.delete(Positive_Feature_train, -1, axis=1)
print('Number of Positive Train: ', Positive_Feature_train.shape)
negative_feature_train_df = df.loc[df['Label'] == 0]
Negative_Feature_train = negative_feature_train_df.values
Negative_Feature_train = np.delete(Negative_Feature_train, -1, axis=1)
print('Number of Negative Train: ', Negative_Feature_train.shape)

positive_feature_test_df = df_test.loc[df_test['Label'] == 1]
Positive_Feature_test = positive_feature_test_df.values
Positive_Feature_test = np.delete(Positive_Feature_test, -1, axis=1)
print('Number of Positive Test: ', Positive_Feature_test.shape)
negative_feature_test_df = df_test.loc[df_test['Label'] == 0]
Negative_Feature_test = negative_feature_test_df.values
Negative_Feature_test = np.delete(Negative_Feature_test, -1, axis=1)
print('Number of Negative Test: ', Negative_Feature_test.shape)

nominal_feature = [11]
data = np.concatenate((Positive_Feature_train, Negative_Feature_train))
num_samples = data.shape[0]
num_features = data.shape[1]
num_bins = 100
bounds = np.zeros((num_bins+1, num_features))
for i in range(num_features):
    if i not in nominal_feature:
        bounds[:, i] = np.histogram(data[:, i], bins=num_bins)[1]


nf_tr = Negative_Feature_train
transfer(nf_tr, bounds, num_features, nominal_feature)
pf_tr = Positive_Feature_train
transfer(pf_tr, bounds, num_features, nominal_feature)
nf_te = Negative_Feature_test
transfer(nf_te, bounds, num_features, nominal_feature)
pf_te = Positive_Feature_test
transfer(pf_te, bounds, num_features, nominal_feature)

np.savez(name.split('.')[0] + '.npz', P_F_tr=pf_tr, P_F_te=pf_te, N_F_tr=nf_tr, N_F_te=nf_te)






