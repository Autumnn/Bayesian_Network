from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np


def view_distribution(data_positive, data_negative, name):
    data = np.concatenate((data_negative, data_positive))
    size = data.shape
    if len(size) > 2:
        print("Input data is nor 2 dimensions matrix")
    else:
        Num = size[1]
        for j in range(Num):
            #MIN = min(data[:, j])
            MIN = 0.0000000001
            #MAX = max(data[:, j])
            MAX = 0.000000001
            Bin = np.linspace(MIN, MAX, 100)
            fig = plt.figure()
            Title = "Histogram of " + name + " the " + str(j) + "th Feature"
            fig.canvas.set_window_title(Title)
            fig.subplots_adjust(hspace=0.4)

            ax = plt.subplot(2,1,1)
            ax.set_title("Positive")
            ax.hist(data_positive[:,j], bins = Bin, facecolor='yellowgreen')

            ax = plt.subplot(2,1,2)
            ax.set_title("Negative")
            ax.hist(data_negative[:,j], bins = Bin, facecolor='blue')

            File_name = "Histogram_of_" + name + "_the_" + str(j) + "th_Feature.png"
            fig.savefig(File_name)

