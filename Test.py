import numpy as np
import xgboost as xgb
from pomegranate import BayesianNetwork


bayes_name = 'High_IR_Data_cross_folder_BayesNet/kr-vs-k-one_vs_fifteen/kr-vs-k-one_vs_fifteen_1_Cross_Folder_Chow-Liu_BayesNet.json'
name = 'High_IR_Data_cross_folder/kr-vs-k-one_vs_fifteen/kr-vs-k-one_vs_fifteen_1_Cross_Folder.npz'
r = np.load(name)

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

bayes = BayesianNetwork.from_json(bayes_name)
pt = bayes.log_probability(Negative_Features_train).sum()
print('Chow-Liu Loss: ', pt)

Negative_Features_train_prob = bayes.probability(Negative_Features_train)
Positive_Features_train_prob = np.zeros((Num_Positive_train, 1))
for i in range(Num_Positive_train):
    try:
        Positive_Features_train_prob[i] = bayes.probability(Positive_Features_train[i])
    except KeyError:
        Positive_Features_train_prob[i] = 0

max_prob = np.max(Positive_Features_train_prob)
print(max_prob)

if max_prob > 0:
    index = np.where(Negative_Features_train_prob <= max_prob)
else:
    index = np.argsort(Negative_Features_train_prob)[0:Num_Positive_train-1]
print(index)


Negative_Features_Filter_train = Negative_Features_train[index]
Num_Negative_Filter_train = Negative_Features_Filter_train.shape[0]
print(Num_Negative_Filter_train)
Negative_Labels_Filter_train = np.linspace(0, 0, Num_Negative_Filter_train)

Feature_train = np.concatenate((Positive_Features_train, Negative_Features_Filter_train))
Label_train = np.concatenate((Positive_Labels_train, Negative_Labels_Filter_train))
Feature_test = np.concatenate((Positive_Features_test, Negative_Features_test))
Label_test = np.concatenate((Positive_Labels_test, Negative_Labels_test))

clf = xgb.XGBClassifier()
clf.fit(Feature_train, Label_train)
Label_predict = clf.predict(Feature_test)
Label_score = clf.predict_proba(Feature_test)

Feature_train_o = np.concatenate((Positive_Features_train, Negative_Features_train))
Label_train_o = np.concatenate((Positive_Labels_train, Negative_Labels_train))
clff = xgb.XGBClassifier()
clff.fit(Feature_train_o, Label_train_o)
Label_predict_f = clff.predict_proba(Feature_test)

Num_test = Num_Positive_test + Num_Negative_test
for i in range(Num_test):
    feature = Feature_test[i]
    try:
        feature_prob = bayes.probability(feature)
    except KeyError:
        feature_prob = 0
    if feature_prob > max_prob:
        Label_predict[i] = 0
        Label_score[i][0],Label_score[i][1] = 1,0
    print(Label_test[i], Label_predict[i], Label_score[i], Label_predict_f[i])

