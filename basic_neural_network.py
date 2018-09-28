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
    A = [None]*3
    W[0] = np.vstack( np.random.rand(nn*feature_size).reshape(nn,feature_size))
    W[0]
    W[1] =  np.random.rand(nn).reshape(1,nn)
    b[0] = np.zeros(nn).reshape(nn,1)
    b[0].shape
    b[1] = np.zeros(1).reshape(1,1)

    A[0] = x
    A[0].shape
    A[1] = np.zeros(nn*nn)

    for j in range(epocs):
        L = 0.0
        accuracy = 0.0
        for i in range(m):
            #Foward propagation first layer
            Z = [None]*2
            a = A[0][i].reshape(np.size(A[0][i],0),1)
            a.shape
            W[0].shape
            Z[0] = np.dot(W[0],a)+ b[0]
            A[1] = tanh(Z[0])

            # g(z) = tanh(z) g' = gp  used in back propegation
            gp = 1 - (A[1]**2)

            #second layer
            Z[1] = np.dot(W[1],A[1]) +b[1]

            A[2] = sigmoid(Z[1])

            L += cost_fuct(A[2],y[i])

            if (A[2] >= .5 and y[i] == 1):
                accuracy+=1.0
            elif( A[2] < .5 and y[i] == 0):
                accuracy+=1.0


            print('Traning acc: ' +str(accuracy/ (i+1.0)))


            #Back propegation
            dZ = [None]*2
            dW = [None]*2
            db = [None]*2

            dZ[1] = A[2] - y[i]
            dW[1] = dZ[1] *A[1].T

            db[1] = np.sum(dZ[1],axis=1,keepdims=True)



            dZ[0] = W[1].T * dZ[1] * gp
            dZ[0].shape

            a.T.shape
            dW[0] = dZ[0] * a.T
            db[0] = np.sum(dZ[0], axis=1, keepdims=True)
            dW[1].shape

            #learing rate and apply update to weights
            alpha = 1
            W[0] = W[0] - alpha*dW[0]
            b[0] = b[0] - alpha*db[0]
            W[0] = W[0] - alpha*dW[0]
            b[0] = b[0] - alpha*db[0]
        print ('cost: ' + str((1.0/m)*L))
        #Validation
        accuracy = 0.0
        for i in range(np.size(x_validation,0)):
                Z = [None]*2
                a = x_validation[i].reshape(np.size(x_validation,1),1)

                Z[0] = np.dot(W[0],a)+ b[0]
                A[1] = tanh(Z[0])

                Z[1] = np.dot(W[1],A[1]) +b[1]

                A[2] = sigmoid(Z[1])

                L += cost_fuct(A[2],y[i])

                Z[1] = np.dot(W[1],A[1]) +b[1]

                A[2] = sigmoid(Z[1])

                if (A[2] >= .5 and y_validation[i] == 1):
                    accuracy+=1.0
                elif( A[2] < .5 and y_validation[i] == 0):
                    accuracy+=1.0

        accuracy /= np.size(x_validation,0)
        print('validation acc: ' +str(accuracy))

def cost_fuct(A,Y):
    loss = -(Y * np.log(A) +(1.0-Y)*np.log(1.0-A))
    return loss

def sigmoid(Z):
    return(1.0/(1.0 + np.exp(-Z)))

def tanh(Z):
    return np.tanh(Z)





if __name__ == '__main__':
    data, data_labels = Get_Data()
    data = data/10.0
    data_labels
    data_labels = fix_labels(data_labels)
    data_labels

    neural_network(data, data_labels,nn = 4, epocs = 1)
