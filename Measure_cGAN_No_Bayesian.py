import os
import numpy as np
import xgboost as xgb
from keras.models import load_model
from imblearn.over_sampling import SMOTE
from metrics_list import metric_list
from pomegranate import BayesianNetwork
import math

#path = "High_IR_Data_cross_folder"
#bayes_path = "High_IR_Data_cross_folder_BayesNet"
#path = "Folds5x2_npz"
#bayes_path = "Folds5x2_BayesNet"
path = "CMAPSSData_npz"
bayes_path = "CMAPSSData_BayesNet"
dirs = os.listdir(path)
# test Git


for Dir in dirs:
    print("Data Set Name: ", Dir)
    dir_path = path + "/" + Dir
    bayes_dir_path = bayes_path + "/" + Dir
    files = os.listdir(dir_path)  # Get files in the folder

    methods = ["xGBoost", "SMOTE", "cGAN", "SMOTE-SMOTE", "cGAN-cGAN",
               "SMOTE-cGAN", "cGAN-SMOTE"]

    for m in methods:
        Num_Cross_Folders = 1
        ml_record = metric_list(np.array([1]), np.array([1]), Num_Cross_Folders)
        i = 0
        for file in files:
            bayes_name = bayes_dir_path + '/' + file.split('.')[0] + '_Chow-Liu_BayesNet.json'
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

            clf = xgb.XGBClassifier()
            if m == "xGBoost":
                Feature_train = Features_train_o
                Label_train = Labels_train_o
                clf.fit(Feature_train, Label_train)
                Label_predict = clf.predict(Feature_test)
                Label_score = clf.predict_proba(Feature_test)
            elif m == "SMOTE":
                sm = SMOTE()
                Feature_train, Label_train = sm.fit_sample(Features_train_o, Labels_train_o)
                clf.fit(Feature_train, Label_train)
                Label_predict = clf.predict(Feature_test)
                Label_score = clf.predict_proba(Feature_test)
            elif m == "cGAN":
                #                input_dim, G_dense, D_dense = cGANStructure.Structure(Dir)  # for UCI data
                input_dim = 70
                G_dense = 90
                D_dense = 45
                Pre_train_epoches = 100
                Train_epoches = 1000
                Model_name = "cGAN_" + Dir + "_folder_" + str(i) + "_G-dense_" + str(G_dense) + "_pretrain_" + str(
                    Pre_train_epoches) + "_D-dense_" + str(D_dense) + "_epoches_" + str(Train_epoches) + ".h5"
                Model_path = "CMAPSSData_cGAN_Model"
                model = load_model(Model_path + "/" + Model_name)
                num_create_samples = Num_Negative_train - Num_Positive_train
                Noise_Input = np.random.uniform(0, 1, size=[num_create_samples, input_dim])
                condition_samples = np.linspace(1, 1, num_create_samples)
                sudo_Samples = model.predict([Noise_Input, condition_samples])
                Feature_train = np.concatenate((Features_train_o, sudo_Samples))
                Label_train = np.concatenate((Labels_train_o, condition_samples))
                clf.fit(Feature_train, Label_train)
                Label_predict = clf.predict(Feature_test)
                Label_score = clf.predict_proba(Feature_test)
            elif m == "SMOTE-SMOTE":
                expand_rate_for_Majority = 0.5
                num_create_samples = int(np.ceil(Num_Negative_train * expand_rate_for_Majority))
                em = SMOTE(ratio={0: num_create_samples + Num_Negative_train})
                Re_Features_o, Labels_o = em.fit_sample(Features_train_o, Labels_train_o)
                sm = SMOTE()
                Feature_train, Label_train = sm.fit_sample(Re_Features_o, Labels_o)
                clf.fit(Feature_train, Label_train)
                Label_predict = clf.predict(Feature_test)
                Label_score = clf.predict_proba(Feature_test)
            elif m == "cGAN-cGAN":
                #                input_dim, G_dense, D_dense = cGANStructure.Structure(Dir)  # for UCI data
                input_dim = 70
                G_dense = 90
                D_dense = 45
                Pre_train_epoches = 100
                Train_epoches = 1000
                Model_name = "cGAN_" + Dir + "_folder_" + str(i) + "_G-dense_" + str(G_dense) + "_pretrain_" + str(
                    Pre_train_epoches) + "_D-dense_" + str(D_dense) + "_epoches_" + str(Train_epoches) + ".h5"
                Model_path = "CMAPSSData_cGAN_Model"
                model = load_model(Model_path + "/" + Model_name)
                expand_rate_for_Majority = 0.5
                num_create_samples = int(np.ceil(Num_Negative_train * expand_rate_for_Majority))
                Noise_Input = np.random.uniform(0, 1, size=[num_create_samples, input_dim])
                condition_samples = np.linspace(0, 0, num_create_samples)
                sudo_Samples = model.predict([Noise_Input, condition_samples])
                Re_Feature_train = np.concatenate((Features_train_o, sudo_Samples))
                Re_Label_train = np.concatenate((Labels_train_o, condition_samples))

                num_create_minority_samples = Num_Negative_train - Num_Positive_train + num_create_samples
                Noise_m_Input = np.random.uniform(0, 1, size=[num_create_minority_samples, input_dim])
                minority_condition_samples = np.linspace(1, 1, num_create_minority_samples)
                minority_sudo_Samples = model.predict([Noise_m_Input, minority_condition_samples])
                Feature_train = np.concatenate((Re_Feature_train, minority_sudo_Samples))
                Label_train = np.concatenate((Re_Label_train, minority_condition_samples))
                clf.fit(Feature_train, Label_train)
                Label_predict = clf.predict(Feature_test)
                Label_score = clf.predict_proba(Feature_test)
            elif m == "SMOTE-cGAN":
                expand_rate_for_Majority = 0.5
                num_create_samples = int(np.ceil(Num_Negative_train * expand_rate_for_Majority))
                em = SMOTE(ratio={0: num_create_samples + Num_Negative_train})
                Re_Features_o, Labels_o = em.fit_sample(Features_train_o, Labels_train_o)
                #                input_dim, G_dense, D_dense = cGANStructure.Structure(Dir)  # for UCI data
                input_dim = 70
                G_dense = 90
                D_dense = 45
                Pre_train_epoches = 100
                Train_epoches = 1000
                Model_name = "cGAN_" + Dir + "_folder_" + str(i) + "_G-dense_" + str(G_dense) + "_pretrain_" + str(
                    Pre_train_epoches) + "_D-dense_" + str(D_dense) + "_epoches_" + str(Train_epoches) + ".h5"
                Model_path = "CMAPSSData_cGAN_Model"
                model = load_model(Model_path + "/" + Model_name)
                num_create_minority_samples = Num_Negative_train - Num_Positive_train + num_create_samples
                Noise_m_Input = np.random.uniform(0, 1, size=[num_create_minority_samples, input_dim])
                minority_condition_samples = np.linspace(1, 1, num_create_minority_samples)
                minority_sudo_Samples = model.predict([Noise_m_Input, minority_condition_samples])
                Feature_train = np.concatenate((Re_Features_o, minority_sudo_Samples))
                Label_train = np.concatenate((Labels_o, minority_condition_samples))
                clf.fit(Feature_train, Label_train)
                Label_predict = clf.predict(Feature_test)
                Label_score = clf.predict_proba(Feature_test)
            elif m == "cGAN-SMOTE":
                #                input_dim, G_dense, D_dense = cGANStructure.Structure(Dir)  # for UCI data
                input_dim = 70
                G_dense = 90
                D_dense = 45
                Pre_train_epoches = 100
                Train_epoches = 1000
                Model_name = "cGAN_" + Dir + "_folder_" + str(i) + "_G-dense_" + str(G_dense) + "_pretrain_" + str(
                    Pre_train_epoches) + "_D-dense_" + str(D_dense) + "_epoches_" + str(Train_epoches) + ".h5"
                Model_path = "CMAPSSData_cGAN_Model"
                model = load_model(Model_path + "/" + Model_name)
                expand_rate_for_Majority = 0.5
                num_create_samples = int(np.ceil(Num_Negative_train * expand_rate_for_Majority))
                Noise_Input = np.random.uniform(0, 1, size=[num_create_samples, input_dim])
                condition_samples = np.linspace(0, 0, num_create_samples)
                sudo_majority_Samples = model.predict([Noise_Input, condition_samples])
                Re_Features_o = np.concatenate((Features_train_o, sudo_majority_Samples))
                Labels_o = np.concatenate((Labels_train_o, condition_samples))
                sm = SMOTE()
                Feature_train, Label_train = sm.fit_sample(Re_Features_o, Labels_o)
                clf.fit(Feature_train, Label_train)
                Label_predict = clf.predict(Feature_test)
                Label_score = clf.predict_proba(Feature_test)

            ml_record.measure(0, 0, i, Label_test, Label_predict)
            ml_record.auc_measure(0, 0, i, Label_test, Label_score[:,1])
            i += 1

        file_wirte = "Result_CMAPSSData_cGAN_no_bayesian_1.txt"
        ml_record.output(file_wirte, m, Dir)