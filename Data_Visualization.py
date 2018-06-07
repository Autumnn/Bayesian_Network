from __future__ import print_function
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import colors as mcolors
import Read_Data as RD
import seaborn as sns


colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

#file = 'shuttle-2_vs_5.dat'
file = 'abalone19.dat'
name = file.split('.')[0]
RD.Initialize_Data(file, has_nominal=True, nominal_index=[0], nominal_value=['M', 'F', 'I'])
print('Number of Positive: ', RD.Num_positive)
print('Number of Negative: ', RD.Num_negative)

#df = pd.DataFrame(RD.Features, columns=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9'])
df = pd.DataFrame(RD.Features, columns=['Sex', 'Length', 'Diameter', 'Height', 'Whole_Weight', 'Shucked_Weight', 'Viscera_Weight', 'Shell_Weight'])
#df['Label'] = pd.Series(RD.Labels, index=df.index)
df['Label'] = RD.Labels

#sns.FacetGrid(df, hue='Label').map(plt.scatter, 'Sex', 'Length')

#ax = sns.boxplot(x='Label', y='Length', data=df)
#ax = sns.stripplot(x='Label', y='Length', data=df, jitter=True)

#sns.violinplot(x='Label', y='Length', data=df)

'''
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



