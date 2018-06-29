import numpy as np
from matplotlib import pyplot as plt
from pomegranate import BayesianNetwork


data_file = 'CMAPSSData_npz/FD001/train_FD001.npz'
bayesnet_file = 'CMAPSSData_BayesNet/FD001/train_FD001_Chow-Liu_BayesNet.json'

r = np.load(data_file)
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
print("Po_tr: ", Num_Positive_train, "Ne_tr: ", Num_Negative_train,
      "Po_te: ", Num_Positive_test, "Ne_te: ", Num_Negative_test)

bayes = BayesianNetwork.from_json(bayesnet_file)
Negative_Features_train_prob = bayes.probability(Negative_Features_train)

Positive_Features_train_prob = np.linspace(0, 0, Num_Positive_train)
for k in range(Num_Positive_train):
    try:
        Positive_Features_train_prob[k] = bayes.probability(Positive_Features_train[k])
    except KeyError:
        Positive_Features_train_prob[k] = 0
#    print(Positive_Features_train_prob[k])

data = Negative_Features_train_prob
data_sorted = np.sort(data)
#for i in data_sorted:
#    print(i)

fig = plt.figure()
Title = "Histogram of Multivariable Probability"
fig.canvas.set_window_title(Title)
fig.subplots_adjust(hspace=0.8)
percentage_to_show = [0.9, 0.2, 0.1]
num_p = len(percentage_to_show)

for i in range(num_p):
    #MIN = min(data_sorted)
    MIN = data_sorted[0]
    #MAX = max(data_sorted)
    num = data_sorted.shape[0]
    num_to_show = int(np.ceil(num*percentage_to_show[i]))
    print(num_to_show)
    MAX = data_sorted[num_to_show]

    Bin = np.linspace(MIN, MAX, 100)
    ax = plt.subplot(num_p+1, 1, i+1)
    ax_title = "Percentage to show " + str(percentage_to_show[i]*100) + "%"
    ax.set_title(ax_title)
    ax.hist(data[0:num_to_show], bins=Bin, facecolor='blue')

ax = plt.subplot(num_p+1, 1, num_p+1)
ax.set_title("Positive")
p = Positive_Features_train_prob.tolist()
p_sorted = np.sort(p)
for i in p_sorted:
    print(i)

# MAX = max(data_sorted)
num = p_sorted.shape[0]
num_to_show = int(np.ceil(num * 0.9))
'''
MIN = p_sorted[num_to_show]
print(MIN)
print(num_to_show)
MAX = p_sorted[-1]
print(MAX)
Bin = np.linspace(MIN, MAX, 100)
'''
ax.hist(p_sorted[num_to_show:], bins=Bin, facecolor='yellowgreen')

File_name = "Histogram_of_Multivaribale_Probability.png"
fig.savefig(File_name)
