import os
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

path = "High_IR_Data_npz"
files= os.listdir(path) #Get files in the folder
for file in files:
    print("File Name: ", file)
    name = file.split(".")[0]
    dir = path + "/" + file
    r = np.load(dir)

    Positive_Features = r["P_F"]
    Num_Positive = Positive_Features.shape[0]
    Positive_Labels = np.linspace(1,1,Num_Positive)
    Negative_Features = r["N_F"]
    Num_Negative = Negative_Features.shape[0]
    Negative_Labels = np.linspace(0,0,Num_Negative)

    Features = np.concatenate((Positive_Features, Negative_Features))
    Labels = np.concatenate((Positive_Labels, Negative_Labels))

    Num_Cross_Folders = 5
    min_max_scalar = preprocessing.MinMaxScaler()
    Re_Features = min_max_scalar.fit_transform(Features)

    #skf = StratifiedKFold(n_splits=Num_Cross_Folders, shuffle=True)
    skf = StratifiedKFold(n_splits=Num_Cross_Folders, shuffle=False)

    i = 0
    for train_index, test_index in skf.split(Re_Features, Labels):
        Feature_train, Feature_test = Re_Features[train_index], Re_Features[test_index]
        Label_train, Label_test = Labels[train_index], Labels[test_index]

        Positive_Feature_train = Feature_train[np.where(Label_train == 1)]
        Positive_Feature_test = Feature_test[np.where(Label_test == 1)]
        Negative_Features_train = Feature_train[np.where(Label_train == 0)]
        Negative_Features_test = Feature_test[np.where(Label_test == 0)]

        saved_name = name + "_" + str(i) + "_Cross_Folder.npz"
        np.savez(saved_name, P_F_tr = Positive_Feature_train, P_F_te = Positive_Feature_test, N_F_tr = Negative_Features_train, N_F_te = Negative_Features_test)

        i += 1