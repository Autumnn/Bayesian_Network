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
    print(item)
    for j in range(20):
        Label[item+j] = 1

df['Label'] = Label
df.drop(columns=[0, 1], inplace=True)

positive_feature_train_df = df.loc[df['Label'] == 1]
Positive_Feature_train = positive_feature_train_df.values
Positive_Feature_train = np.delete(Positive_Feature_train, -1, axis=1)
print('Number of Positive: ', Positive_Feature_train.shape)
negative_feature_train_df = df.loc[df['Label'] == 0]
#negative_feature_train_df.drop(columns=['Label'], inplace=True)
Negative_Feature_train = negative_feature_train_df.values
Negative_Feature_train = np.delete(Negative_Feature_train, -1, axis=1)
print('Number of Negative: ', Negative_Feature_train.shape)


data = np.concatenate((Positive_Feature_train, Negative_Feature_train))
num_samples = data.shape[0]
num_features = data.shape[1]
num_bins = 100
bounds = np.zeros((num_bins+1, num_features))
for i in range(num_features):
    if i not in nominal_feature:
        bounds[:, i] = np.histogram(data[:, i], bins=num_bins)[1]

nf = RD.get_negative_feature()
transfer(nf, bounds, num_features, nominal_feature)
pf = RD.get_positive_feature()
transfer(pf, bounds, num_features, nominal_feature)
np.savez(name+'.npz', N_F=nf, P_F=pf)

'''
bayes = BayesianNetwork.from_samples(nf, algorithm='exact-dp')
pt = bayes.log_probability(nf).sum()
print('Exact Shortest:', pt)

bayes = BayesianNetwork.from_samples(nf, algorithm='exact')
pt = bayes.log_probability(nf).sum()
print('Exact A*', pt)

bayes = BayesianNetwork.from_samples(nf, algorithm='greedy')
pt = bayes.log_probability(nf).sum()
print('Greedy', pt)


bayes = BayesianNetwork.from_samples(nf, algorithm='chow-liu')
pt = bayes.log_probability(nf).sum()
print('Chow-Liu', pt)

with open(name+'_bayes.json', 'w') as w:
    w.write(bayes.to_json())
'''






