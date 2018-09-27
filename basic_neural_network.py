import os
import time
import numpy as np
import matplotlib.pyplot as plt

#*****************************************************************************
#Getting raw data
#*****************************************************************************
#Working with
def Get_Data():
    fname = 'breast-cancer-wisconsin.data'
    data = np.loadtxt(fname, delimiter=',') #Removed instances of missing data
    data = data.reshape(683,11)
    data_labels = data[:,(10)]
    data_labels[4]
    data = data[:,(1,2,3,4,5,6,7,8,9)]
    return data, data_labels


#labels or output should be binary 0 or 1
def fix_labels(y):

    for i in range(np.size(y,0)):
        if y[i] == 2:
            y[i] = 0.0
        else:
            y[i] = 1
    return y


def neural_network(data,data_labels, nn, epocs ):
    nn=4
    n = np.size(data,0)
    feature_size = np.size(data,1)
    testing_size = n-int(n*.1)

    x = data[0:n-int(n*.1),]
    y = data_labels[0:testing_size,].reshape(testing_size,1)

    m = np.size(x,0)

    x_validation = data[testing_size: n,]
    y_validation = data_labels[testing_size: n,]


    W = [None]*2
    W
    b = [None]*2
    a = [None]*2
    W[0] = np.vstack( np.random.rand(nn*feature_size).reshape(nn,feature_size))
    W[1] =  np.random.rand(nn).reshape(1,nn)
    b[0] = np.zeros(nn)
    b[1] = np.zeros(1)

    a[0] = x
    a[1] = np.zeros(nn)

    for i in range(m):
        Z = [None]*2
        




if __name__ == '__main__':
    data, data_labels = Get_Data()
    data_labels
    data_labels = fix_labels(data_labels)
    data_labels

    neural_network(data, data_labels,nn = 4, epocs = 1)
