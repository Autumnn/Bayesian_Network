import numpy as np
import xgboost as xgb
from pomegranate import BayesianNetwork
import pandas as pd
from matplotlib import colors as mcolors
import seaborn as sns


name = 'Folds5x2.xlsx'
df = pd.read_excel(name, index_col=None)
num_samples = len(df)
print(num_samples)