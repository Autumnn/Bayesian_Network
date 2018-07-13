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
path = "Folds5x2_npz"
bayes_path = "Folds5x2_BayesNet"
dirs = os.listdir(path)
# test Git


for Dir in dirs:
    print("Data Set Name: ", Dir)
    dir_path = path + "/" + Dir
    bayes_dir_path = bayes_path + "/" + Dir
    files = os.listdir(dir_path)  # Get files in the folder

    methods = ["xGBoost", "SMOTE", "cGAN", "SMOTE-SMOTE", "cGAN-cGAN",
               "SMOTE-cGAN", "cGAN-SMOTE", "Bayesian", "Bayesian_e", "Pure_Bayesian"]

    for m in methods:
        Num_Cross_Folders = 5
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
                Model_path = "Folds5x2_cGAN_Model"
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
                Model_path = "Folds5x2_cGAN_Model"
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
                Model_path = "Folds5x2_cGAN_Model"
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
                Model_path = "Folds5x2_cGAN_Model"
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
            elif m == "Bayesian":
                bayes = BayesianNetwork.from_json(bayes_name)
                Negative_Features_train_prob = bayes.probability(Negative_Features_train)
                Positive_Features_train_prob = np.zeros((Num_Positive_train, 1))
                for k in range(Num_Positive_train):
                    try:
                        Positive_Features_train_prob[k] = bayes.probability(Positive_Features_train[k])
                    except KeyError:
                        Positive_Features_train_prob[k] = 0

                max_prob = np.max(Positive_Features_train_prob)
                print(max_prob)

                if max_prob > 0:
                    index = np.where(Negative_Features_train_prob <= max_prob)
                else:
                    index = np.argsort(Negative_Features_train_prob)[0:Num_Positive_train - 1]

                Negative_Features_Filter_train = Negative_Features_train[index]
                Num_Negative_Filter_train = Negative_Features_Filter_train.shape[0]
                Negative_Labels_Filter_train = np.linspace(0, 0, Num_Negative_Filter_train)
                Feature_train = np.concatenate((Positive_Features_train, Negative_Features_Filter_train))
                Label_train = np.concatenate((Positive_Labels_train, Negative_Labels_Filter_train))
                clf.fit(Feature_train, Label_train)
                Label_predict = clf.predict(Feature_test)
                Label_score = clf.predict_proba(Feature_test)
                Num_test = Num_Positive_test + Num_Negative_test
                for j in range(Num_test):
                    feature = Feature_test[j]
                    try:
                        feature_prob = bayes.probability(feature)
                    except KeyError:
                        feature_prob = 0
                    if feature_prob > max_prob:
                        Label_predict[j] = 0
                        Label_score[j] = [1, 0]
            elif m == "Bayesian_e":
                bayes = BayesianNetwork.from_json(bayes_name)
                Negative_Features_train_prob = bayes.probability(Negative_Features_train)
                Positive_Features_train_prob = np.zeros((Num_Positive_train, 1))
                for k in range(Num_Positive_train):
                    try:
                        Positive_Features_train_prob[k] = bayes.probability(Positive_Features_train[k])
                    except KeyError:
                        Positive_Features_train_prob[k] = 0

                max_prob = np.max(Positive_Features_train_prob)
                print(max_prob)

                if max_prob > 0:
                    index = np.where(Negative_Features_train_prob <= max_prob)
                    if len(index) < Num_Positive_train:
                        index = np.argsort(Negative_Features_train_prob)[0:Num_Positive_train - 1]
                else:
                    index = np.argsort(Negative_Features_train_prob)[0:Num_Positive_train - 1]

                Negative_Features_Filter_train = Negative_Features_train[index]
                Num_Negative_Filter_train = Negative_Features_Filter_train.shape[0]
                Negative_Labels_Filter_train = np.linspace(0, 0, Num_Negative_Filter_train)
                Feature_train = np.concatenate((Positive_Features_train, Negative_Features_Filter_train))
                Label_train = np.concatenate((Positive_Labels_train, Negative_Labels_Filter_train))
                clf.fit(Feature_train, Label_train)
                Label_predict = clf.predict(Feature_test)
                Label_score = clf.predict_proba(Feature_test)
                Num_test = Num_Positive_test + Num_Negative_test
                for j in range(Num_test):
                    feature = Feature_test[j]
                    try:
                        feature_prob = bayes.probability(feature)
                    except KeyError:
                        feature_prob = 0
                    if feature_prob > max_prob:
                        Label_predict[j] = 0
                        Label_score[j] = [1, 0]
            elif m == "Pure_Bayesian":
                bayes = BayesianNetwork.from_json(bayes_name)
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
                        Label_score[k][0] = 0  # Probability of being Negative
                        Label_score[k][1] = 1  # Probability of being Positive
                        Label_predict[k] = 1
                    else:
                        z = np.max(index_array)
                        pr = z / Num_test
                        Label_score[k][0] = 1 - math.exp(-69.315 * pr)  # pr < 1% --> Probability of being Negative < 50%
                        Label_score[k][1] = 1 - Label_score[k][0]
                        if Label_score[k][0] < 0.5:
                            Label_predict[k] = 1

            ml_record.measure(0, 0, i, Label_test, Label_predict)
            ml_record.auc_measure(0, 0, i, Label_test, Label_score[:,1])
            i += 1

        file_wirte = "Result_Folds5x2_cGAN_all.txt"
        ml_record.output(file_wirte, m, Dir)