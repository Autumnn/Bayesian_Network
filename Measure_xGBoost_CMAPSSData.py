import os
import numpy as np
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from metrics_list import metric_list

path = "CMAPSSData_npz"
dirs = os.listdir(path) #Get files in the folder

for Dir in dirs:
    print("Data Set Name: ", Dir)
    dir_path = path + "/" + Dir
    files = os.listdir(dir_path)  # Get files in the folder

    methods = ["xGBoost", "SMOTE", "SMOTE-SMOTE"]
    for m in methods:
        Num_Cross_Folders = 1
        ml_record = metric_list(np.array([1]), np.array([1]), Num_Cross_Folders)
        i = 0
        for file in files:
            name = dir_path + '/' + file
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

            print(i, " folder; ", "Po_tr: ", Num_Positive_train, "Ne_tr: ", Num_Negative_train,
                  "Po_te: ", Num_Positive_test, "Ne_te: ", Num_Negative_test)

            Features_train_o = np.concatenate((Positive_Features_train, Negative_Features_train))
            Labels_train_o = np.concatenate((Positive_Labels_train, Negative_Labels_train))
            #                print(Labels_train_o)
            Feature_test = np.concatenate((Positive_Features_test, Negative_Features_test))
            Label_test = np.concatenate((Positive_Labels_test, Negative_Labels_test))
            #                print(Label_test)

            if m == "xGBoost":
                Feature_train = Features_train_o
                Label_train = Labels_train_o
            elif m == "SMOTE":
                sm = SMOTE()
                Feature_train, Label_train = sm.fit_sample(Features_train_o, Labels_train_o)
            elif m == "SMOTE-SMOTE":
                expand_rate_for_Majority = 0.5
                num_create_samples = int(np.ceil(Num_Negative_train * expand_rate_for_Majority))
                em = SMOTE(ratio={0: num_create_samples + Num_Negative_train})
                Re_Features_o, Labels_o = em.fit_sample(Features_train_o, Labels_train_o)
                sm = SMOTE()
                Feature_train, Label_train = sm.fit_sample(Re_Features_o, Labels_o)

            clf = xgb.XGBClassifier()
            clf.fit(Feature_train, Label_train)
            Label_predict = clf.predict(Feature_test)
            ml_record.measure(0, 0, i, Label_test, Label_predict)
            Label_score = clf.predict_proba(Feature_test)
            ml_record.auc_measure(0, 0, i, Label_test, Label_score[:,1])

            i += 1

        file_wirte = "CMAPSSData_xGBoost.txt"
        ml_record.output(file_wirte, m, Dir)