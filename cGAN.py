from keras import optimizers
from keras.layers import Input, Dense, Activation, concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from tqdm import tqdm_notebook as tqdm
from ipywidgets import IntProgress
import numpy as np

def get_generative(G_in, C_in, dense_dim=160, out_dim=10, lr=1e-3):
    x = concatenate([G_in, C_in])
    x = Dense(dense_dim)(x)
    x = Activation('relu')(x)
    G_out = Dense(out_dim, activation='sigmoid')(x)
    G = Model([G_in, C_in], G_out)
    opt = SGD(lr=lr)
#    opt = Adam(lr=lr)
#    G.compile(loss='categorical_crossentropy', optimizer=opt)
#    G.compile(loss='binary_crossentropy', optimizer=opt)
    G.compile(loss='mean_squared_error', optimizer=opt)
    return G, G_out

#G_in = Input(shape=[6])
#G, G_out = get_generative(G_in)
#G.summary()

def get_discriminative(D_in, C_in, dense_dim = 80, lr=1e-3):
    x = concatenate([D_in, C_in])
    x = Dense(dense_dim)(x)
    x = Activation('relu')(x)
    D_out = Dense(1, activation='sigmoid')(x)
    D = Model([D_in, C_in], D_out)
#    dopt = SGD(lr=lr)
    dopt = Adam(lr=lr)
    D.compile(loss='binary_crossentropy', optimizer=dopt)
    return D, D_out

#D_in = Input(shape=[10])
#D, D_out = get_discriminative(D_in)
#D.summary()


def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

def make_gan(GAN_in, C_in, G, D):
    set_trainability(D, False)
    x = G([GAN_in, C_in])
    GAN_out = D([x, C_in])
    GAN = Model([GAN_in, C_in], GAN_out)
#    GAN.compile(loss='mean_squared_error', optimizer=G.optimizer)
    GAN.compile(loss='binary_crossentropy', optimizer=G.optimizer)
    return GAN, GAN_out

#GAN_in = Input([10])
#GAN, GAN_out = make_gan(GAN_in, G, D)
#GAN.summary()

def sample_data(samples):   # ACtually no use, just for test --> it can generate random fake samples
    Number_of_Feature = samples.shape[1]
#    Number_of_Samples = samples.shape[0]
    size = list(samples.shape)
#    print(size)
    Fake_Sample = np.random.rand(size[0],size[1])
    for i in range(Number_of_Feature):
        Min = min(samples[:,i])
        Dis = max(samples[:,i]) - Min
        print("i=", i, " Min=", Min, " Dis", Dis)
        Fake_Sample[:,i] = Fake_Sample[:,i] * Dis + Min
    return Fake_Sample

def sample_data_and_gen(G, C, samples, noise_dim = 6):
    #XT = sample_data(samples)
    XT = samples
    size = list(samples.shape)
    XN_noise = np.random.uniform(0, 1, size=[size[0], noise_dim])
    #print("XN_noise:")
    #print(XN_noise[0])
    #print(XN_noise[-1])
    XN = G.predict([XN_noise, C])
    #print("XN:")
    #print(XN[0])
    #print(XN[-1])
    X = np.concatenate((XT, XN))
    y = np.zeros((2*size[0], 1))
    y[:size[0]] = 1
#    y[size[0]:] = 0
    return X, y

def pretrain(G, D, C_samples, samples, noise_dim = 6, batch_size=64, epoches = 1000):
    X, y = sample_data_and_gen(G, C_samples, samples, noise_dim=noise_dim)
    set_trainability(D, True)
    cc = np.concatenate((C_samples, C_samples))
    D.fit([X, cc], y, epochs=epoches, batch_size=batch_size)

def sample_noise(G, samples, noise_dim=6):
    size = list(samples.shape)
    X = np.random.uniform(0, 1, size=[size[0], noise_dim])
    y = np.ones((size[0], 1))
#    y = np.zeros((size[0], 2))
#    y[:, 1] = 1
    return X, y


def train(GAN, G, D, C_samples, samples, epochs=1000, noise_dim=6, batch_size=64, verbose=False, v_freq=50):
    d_loss = []
    g_loss = []
    e_range = range(epochs)
    if verbose:
        e_range = tqdm(e_range)
    for epoch in e_range:
        X, y = sample_data_and_gen(G, C_samples, samples, noise_dim=noise_dim)
        set_trainability(D, True)
        cc = np.concatenate((C_samples, C_samples))
        d_loss.append(D.train_on_batch([X, cc], y))

        X, y = sample_noise(G, samples, noise_dim=noise_dim)
        set_trainability(D, False)
        g_loss.append(GAN.train_on_batch([X, C_samples], y))
        if verbose and (epoch + 1) % v_freq == 0:
            print("Epoch #{}: Generative Loss: {}, Discriminative Loss: {}".format(epoch + 1, g_loss[-1], d_loss[-1]))
    return d_loss, g_loss
