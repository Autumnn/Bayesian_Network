import pandas as pd
import numpy as np
import Read_Data as RD
from pomegranate import BayesianNetwork


def transfer(data, bounds, num_features, nominal_feature):
    size = data.shape[0]
    for k in range(size):
        initial_sample = data[k, :]
        z = np.linspace(0, 0, num_features)
        for ii in range(num_features):
            if ii not in nominal_feature:
                z[ii] = np.max(np.where(bounds[:, ii] <= initial_sample[ii])[0])
                if z[ii] > 99:
                    z[ii] -= 1
            else:
                z[ii] = initial_sample[ii]
        data[k, :] = z


file = 'High_IR_Data/shuttle-2_vs_5.dat'

name = file.split('.')[0]
print(name)


RD.Initialize_Data(file)

print('Number of Positive: ', RD.Num_positive)
print('Number of Negative: ', RD.Num_negative)

nominal_feature = []
data = RD.get_feature()
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

bayes = BayesianNetwork.from_samples(nf, algorithm='exact-dp')
pt = bayes.log_probability(nf).sum()
print('Exact Shortest', pt)

'''
bayes = BayesianNetwork.from_samples(nf, algorithm='exact-dp')
pt = bayes.log_probability(nf).sum()
print('Exact Shortest:', pt)

bayes = BayesianNetwork.from_samples(nf, algorithm='exact')     # Error
pt = bayes.log_probability(nf).sum()
print('Exact A*', pt)

bayes = BayesianNetwork.from_samples(nf, algorithm='greedy')
pt = bayes.log_probability(nf).sum()
print('Greedy', pt)

bayes = BayesianNetwork.from_samples(nf, algorithm='chow-liu')
pt = bayes.log_probability(nf).sum()
print('Chow-Liu', pt)
'''

with open(name+'_bayes_exact_shortest.json', 'w') as w:
    w.write(bayes.to_json())




