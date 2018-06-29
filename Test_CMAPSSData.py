import numpy as np
from matplotlib import pyplot as plt
import xgboost as xgb
from pomegranate import BayesianNetwork
import math
from sklearn import metrics


data_file = 'CMAPSSData_npz/FD001/train_FD001.npz'
bayesnet_file = 'CMAPSSData_BayesNet/FD001/train_FD001_Chow-Liu_BayesNet.json'

r = np.load(data_file)
Positive_Features_train = r["P_F_tr"]
Num_Positive_train = Positive_Features_train.shape[0]
Positive_Labels_train = np.linspace(1, 1, Num_Positive_train)
Positive_Features_test = r["P_F_te"]
Num_Positive_test = Positive_Features_test.shape[0]
Positive_Labels_test = np.linspace(1, 1, Num_Positive_test)
Negative_Features_train = r["N_F_tr"]
Num_Negative_train = Negative_Features_train.shape[0]
Negative_Labels_train = np.linspace(0, 0, Num_Negative_train)
Negative_Features_test = r["N_F_te"]
Num_Negative_test = Negative_Features_test.shape[0]
Negative_Labels_test = np.linspace(0, 0, Num_Negative_test)
print("Po_tr: ", Num_Positive_train, "Ne_tr: ", Num_Negative_train,
      "Po_te: ", Num_Positive_test, "Ne_te: ", Num_Negative_test)

Feature_test = np.concatenate((Positive_Features_test, Negative_Features_test))
Label_test = np.concatenate((Positive_Labels_test, Negative_Labels_test))

'''
Features_train_o = np.concatenate((Positive_Features_train, Negative_Features_train))
Labels_train_o = np.concatenate((Positive_Labels_train, Negative_Labels_train))
Feature_train = Features_train_o
Label_train = Labels_train_o
clf = xgb.XGBClassifier()
clf.fit(Feature_train, Label_train)
Label_predict = clf.predict(Feature_test)
Label_score = clf.predict_proba(Feature_test)

print(Label_score.shape)
print(Label_predict[0:5], Label_score[0:5])

'''

bayes = BayesianNetwork.from_json(bayesnet_file)
Negative_Features_train_prob = bayes.probability(Negative_Features_train)
ref_neg_prob = np.sort(Negative_Features_train_prob)

Num_test = Num_Positive_test + Num_Negative_test
Label_predict = np.linspace(0, 0, Num_test)
Label_score = np.zeros((Num_test, 2))
for k in range(Num_test):
    try:
        test_prob = bayes.probability(Feature_test[k])
    except KeyError:
        test_prob = 0

    index_array = np.where(ref_neg_prob <= test_prob)[0]
    if index_array.shape[0] == 0:
        Label_score[k][0] = 0           # Probability of being Negative
        Label_score[k][1] = 1           # Probability of being Positive
        Label_predict[k] = 1
    else:
        z = np.max(index_array)
        pr = z/Num_test
        Label_score[k][0] = 1 - math.exp(-69.315*pr)    # pr < 1% --> Probability of being Negative < 50%
        Label_score[k][1] = 1 - Label_score[k][0]
        if Label_score[k][0] < 0.5:
            Label_predict[k] = 1

    #print(Label_score[k], Label_predict[k])
cf = metrics.confusion_matrix(Label_test, Label_predict)
print(cf)
    
        




