import numpy as np
import os
from pomegranate import BayesianNetwork


#path = "High_IR_Data_cross_folder"
path = "Folds5x2_npz"
dirs = os.listdir(path) #Get files in the folder

for Dir in dirs:
    print("Data Set Name: ", Dir)
    dir_path = path + "/" + Dir
    files = os.listdir(dir_path)  # Get files in the folder

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

        bayes = BayesianNetwork.from_samples(Negative_Features_train, algorithm='chow-liu')
        pt = bayes.log_probability(Negative_Features_train).sum()
        print('Chow-Liu', pt)

        file_name = file.split('.')[0]
        with open(file_name + '_Chow-Liu_BayesNet.json', 'w') as w:
            w.write(bayes.to_json())








