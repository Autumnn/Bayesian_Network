import numpy as np
from keras.layers import Input
import cGAN as gan
import os

path = "CMAPSSData_npz"
dirs = os.listdir(path) #Get files in the folder
First_line = True

for Dir in dirs:
    print("Data Set Name: ", Dir)
    dir_path = path + "/" + Dir
    files= os.listdir(dir_path) #Get files in the folder

    i = 0
    for file in files:
        name = dir_path + '/' + file
        r = np.load(name)

        Positive_Features_train = r["P_F_tr"]
        Num_Positive_train = Positive_Features_train.shape[0]
        Positive_Labels_train = np.linspace(1, 1, Num_Positive_train)

        Negative_Features_train = r["N_F_tr"]
        Num_Negative_train = Negative_Features_train.shape[0]
        Negative_Labels_train = np.linspace(0, 0, Num_Negative_train)

        print(i, " folder; ", "Po_tr: ", Num_Positive_train, "N_F_tr: ", Num_Negative_train)

        Features_train_o = np.concatenate((Positive_Features_train, Negative_Features_train))
        Labels_train_o = np.concatenate((Positive_Labels_train, Negative_Labels_train))

        Num_Features = Positive_Features_train.shape[1]

        input_dim = 70
        G_dense = 90
        D_dense = 45

        print('Generate Models')
        G_in = Input(shape=[input_dim])
        C_in = Input(shape=[1])
        G, G_out = gan.get_generative(G_in, C_in, dense_dim=G_dense, out_dim=Num_Features)
        G.summary()
        D_in = Input(shape=[Num_Features])
        D, D_out = gan.get_discriminative(D_in, C_in, dense_dim=D_dense)
        D.summary()
        GAN_in = Input([input_dim])
        GAN, GAN_out = gan.make_gan(GAN_in, C_in, G, D)
        GAN.summary()

        Pre_train_epoches = 100
        Train_epoches = 10000
        gan.pretrain(G, D, Labels_train_o, Features_train_o, noise_dim=input_dim, epoches=Pre_train_epoches)
        d_loss, g_loss = gan.train(GAN, G, D, Labels_train_o, Features_train_o, epochs=Train_epoches, noise_dim=input_dim,
                                   verbose=True)
        Model_name = "cGAN_" + Dir + "_folder_" + str(i) + "_G-dense_" + str(G_dense) + "_pretrain_" + str(
            Pre_train_epoches) + "_D-dense_" + str(D_dense) + "_epoches_" + str(Train_epoches) + ".h5"
        G.save(Model_name)
        i += 1