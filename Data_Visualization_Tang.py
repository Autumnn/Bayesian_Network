import numpy as np
import xgboost as xgb
from pomegranate import BayesianNetwork
import pandas as pd
from matplotlib import colors as mcolors
import seaborn as sns


colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)       # '#539caf' is a good color !

name = 'Tang.csv'
df = pd.read_csv(name, index_col=None)
num_samples = len(df)

'''
Label = np.linspace(0, 0, num_samples)
unit_list = df['PE'].tolist()
for i in range(num_samples):
    if unit_list[i] < 430.8:
        Label[i] = 2       # window length is 10
    elif 430.8 <= unit_list[i] < 433.4:
        Label[i] = 1
df['Label'] = Label
#df.drop(columns=[0, 1], inplace=True)
'''

attribute_used = list(df)
print(attribute_used)
attribute_used.remove('Y')
print(attribute_used)
sp = sns.pairplot(df, hue='Y', size=3, palette=['#539caf' , colors["darkmagenta"], colors["crimson"]], vars=attribute_used)
sp.savefig(name.split('.')[0] + '_Compare.png')
