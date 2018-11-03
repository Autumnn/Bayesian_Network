import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


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

name = 'Tang.csv'
df = pd.read_csv(name, index_col=None)
num_samples = len(df)

positive_feature_df = df.loc[df['Y'] == 1]
Positive_Feature = positive_feature_df.values
Positive_Feature = np.delete(Positive_Feature, -1, axis=1)
print('Number of Positive: ', Positive_Feature.shape)
Num_Positive = Positive_Feature.shape[0]
Positive_Labels = np.linspace(1, 1, Num_Positive)

negative_feature_df = df.loc[df['Y'] == 0]
Negative_Feature = negative_feature_df.values
Negative_Feature = np.delete(Negative_Feature, -1, axis=1)
print('Number of Negative: ', Negative_Feature.shape)
Num_Negative = Negative_Feature.shape[0]
Negative_Labels = np.linspace(0, 0, Num_Negative)

'''
nominal_feature = []
data = np.concatenate((Positive_Feature, Negative_Feature))
num_samples = data.shape[0]
num_features = data.shape[1]
num_bins = 100
bounds = np.zeros((num_bins+1, num_features))
for i in range(num_features):
    if i not in nominal_feature:
        bounds[:, i] = np.histogram(data[:, i], bins=num_bins)[1]

nf_tr = Negative_Feature
transfer(nf_tr, bounds, num_features, nominal_feature)
pf_tr = Positive_Feature
transfer(pf_tr, bounds, num_features, nominal_feature)
'''

#Features = np.concatenate((pf_tr, nf_tr))
Features = np.concatenate((Positive_Feature, Negative_Feature))
Labels = np.concatenate((Positive_Labels, Negative_Labels))

Num_Cross_Folders = 5
skf = StratifiedKFold(n_splits=Num_Cross_Folders, shuffle=False)

i = 0
for train_index, test_index in skf.split(Features, Labels):
    Feature_train, Feature_test = Features[train_index], Features[test_index]
    Label_train, Label_test = Labels[train_index], Labels[test_index]

    Positive_Feature_train = Feature_train[np.where(Label_train == 1)]
    Positive_Feature_test = Feature_test[np.where(Label_test == 1)]
    Negative_Features_train = Feature_train[np.where(Label_train == 0)]
    Negative_Features_test = Feature_test[np.where(Label_test == 0)]

    saved_name = name.split('.')[0] + "_" + str(i) + "_Cross_Folder.npz"
#    np.savez(saved_name, P_F_tr=Positive_Feature_train, P_F_te=Positive_Feature_test, N_F_tr=Negative_Features_train,
#             N_F_te=Negative_Features_test)
    np.savez(saved_name, F_tr = Feature_train, L_tr = Label_train, F_te = Feature_test, L_te = Label_test)

    i += 1







