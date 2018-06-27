import numpy as np
import os
from imblearn.over_sampling import SMOTE
from pomegranate import BayesianNetwork


path = "High_IR_Data_cross_folder"
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

        Negative_Features_train = r["N_F_tr"]
        Num_Negative_train = Negative_Features_train.shape[0]
        Negative_Labels_train = np.linspace(0, 0, Num_Negative_train)
        print(Num_Negative_train)

        Features_train_o = np.concatenate((Positive_Features_train, Negative_Features_train))
        Labels_train_o = np.concatenate((Positive_Labels_train, Negative_Labels_train))

        expand_rate = 1
        num_create_samples = int(np.ceil(Num_Negative_train * expand_rate))
        em = SMOTE(ratio={0: num_create_samples + Num_Negative_train})
        Re_Features, Re_Labels = em.fit_sample(Features_train_o, Labels_train_o)
        Re_Negative_Features = Re_Features[Num_Positive_train:]
        print(Re_Negative_Features.shape[0])

        bayes = BayesianNetwork.from_samples(Re_Negative_Features, algorithm='chow-liu')
        pt = bayes.log_probability(Negative_Features_train).sum()
        print('Chow-Liu', pt)

        file_name = file.split('.')[0]
        with open(file_name + '_Chow-Liu_BayesNet_SMOTE.json', 'w') as w:
            w.write(bayes.to_json())








