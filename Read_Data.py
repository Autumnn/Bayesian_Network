from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np


def Initialize_Data(dir, has_nominal=False, nominal_index=[0], nominal_value=[]):
    Num_lines = len(open(dir, 'r').readlines())
    num_columns = 0
    data_info_lines = 0
    with open(dir, "r") as get_info:
        print("name", get_info.name)
        for line in get_info:
            if line.find("@") == 0:
                data_info_lines += 1
            else:
                columns = line.split(",")
                num_columns = len(columns)
                break

    global Num_Samples
    Num_Samples = Num_lines - data_info_lines
    print(Num_Samples)
    global Num_Features
    Num_Features = num_columns - 1

    global Features
    Features = np.ones((Num_Samples, Num_Features))
    global Labels
    Labels = np.ones((Num_Samples, 1))

    global Num_positive
    Num_positive = 0
    global Num_negative
    Num_negative = 0

    with open(dir, "r") as data_file:
        print("Read Data", data_file.name)
        l = 0
        for line in data_file:
            l += 1
            if l > data_info_lines:
                # print(line)
                row = line.split(",")
                length_row = len(row)
                # print('Row length',length_row)
                # print(row[0])
                for i in range(length_row):
                    if i < length_row - 1:
                        if has_nominal:
                            if i in nominal_index:
                                Features[l - data_info_lines - 1][i] = nominal_value.index(row[i])
                            else:
                                Features[l - data_info_lines - 1][i] = row[i]
                        else:
                            Features[l - data_info_lines - 1][i] = row[i]
                        # print(Features[l-14][i])
                    else:
                        attri = row[i].strip()
                        # print(attri)
                        if attri == 'negative':
                            Labels[l - data_info_lines - 1][0] = 0
                            Num_negative += 1
                            # print(Labels[l-14][0])
                        else:
                            Labels[l - data_info_lines - 1][0] = 1
                            Num_positive += 1

#    print("Number of Positive: ", Num_positive)
    global Positive_Feature
    Positive_Feature = np.ones((Num_positive, Num_Features))
#    print("Num of Negative: ", Num_negative)
    global Negative_Feature
    Negative_Feature = np.ones((Num_negative, Num_Features))
    index_positive = 0
    index_negative = 0

    for i in range(Num_Samples):
        if Labels[i] == 1:
            Positive_Feature[index_positive] = Features[i]
            index_positive += 1
        else:
            Negative_Feature[index_negative] = Features[i]
            index_negative += 1

    print("Read Completed")

def get_feature():
    return Features

def get_label():
    return  Labels

def get_positive_feature():
    return Positive_Feature

def get_negative_feature():
    return Negative_Feature



