import pandas as pd
import numpy as np
import Read_Data as RD
from pomegranate import BayesianNetwork
import View_Distribution as vd

file = 'abalone19.dat'
name = file.split('.')[0]
print(name)
bayes = BayesianNetwork.from_json(name+'_bayes.json')

r = np.load(name+'.npz')
nf = r['N_F']
pf = r['P_F']

prob_nf = bayes.probability(nf)

num_pf = pf.shape[0]
num_of_col = pf.shape[1]
prob_pf = np.zeros((num_pf, 1))
for i in range(num_pf):
    for j in range(num_of_col):
        if pf[i, j] not in np.unique(nf[:,j]):
#            print(i, j)
            break
        elif j == num_of_col-1:
            prob_pf[i] = bayes.probability(pf[i,:])

for i in prob_pf:
    print(i)
prob_nf = np.array([prob_nf]).reshape(-1, 1)

vd.view_distribution(prob_pf, prob_nf, name)



