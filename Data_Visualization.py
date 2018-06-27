from __future__ import print_function
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import colors as mcolors
import Read_Data as RD
import seaborn as sns
import os


colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

path = "High_IR_Data_npz"
files = os.listdir(path)
for file in files:
    print("Data Set Name: ", file)
    name = path + '/' + file
    r = np.load(name)

    Positive_Features = r["P_F"]
    Num_Positive = Positive_Features.shape[0]
    Positive_Labels = np.linspace(1,1,Num_Positive)
    Negative_Features = r["N_F"]
    Num_Negative = Negative_Features.shape[0]
    Negative_Labels = np.linspace(0,0,Num_Negative)

    Features = np.concatenate((Positive_Features, Negative_Features))
    Labels = np.concatenate((Positive_Labels, Negative_Labels))
    df = pd.DataFrame(Features)
    df['Label'] = Labels

    attribute_used = list(df)
    print(attribute_used)
    attribute_used.remove('Label')
    print(attribute_used)
    sp = sns.pairplot(df, hue='Label', size=3, palette=['#539caf', colors["darkmagenta"]], vars=attribute_used)
    sp.savefig(name+'_Compare.png')

'''
df.drop('Label', axis=1, inplace=True)
sns.heatmap(df.astype(float).corr(), cmap=plt.cm.RdBu, vmax=1.0, linewidths=0.1, linecolor='white', square=True, annot=True)

plt.show()
'''



