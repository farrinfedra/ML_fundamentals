import cvxopt as cvx
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.spatial.distance as dt
import pandas as pd

data_set_images = np.genfromtxt("hw06_images.csv", delimiter = ",")
data_set_labels = np.genfromtxt("hw06_labels.csv", delimiter = "\n")

# training dataset
x_train = data_set_images[0:1000,:]
y_train = data_set_labels[0:1000].astype(int)

#testing dataset
x_test = data_set_images[1000:5000, :]
y_test = data_set_labels[1000:5000].astype(int)


N = x_train.shape[0]
D = x_train.shape[1]
K = np.max(y_train)
N_test = x_test.shape[0]

y_train1 = np.negative(np.ones((N))).astype(int)
y_train2 = np.negative(np.ones((N))).astype(int)
y_train3 = np.negative(np.ones((N))).astype(int)
y_train4 = np.negative(np.ones((N))).astype(int)
y_train5 = np.negative(np.ones((N))).astype(int)

for i in range(N):
    if y_train[i] == 1:
        y_train1[i] = 1
    if y_train[i] == 2:
        y_train2[i] = 1
    if y_train[i] == 3:
        y_train3[i] = 1
    if y_train[i] == 4:
        y_train4[i] = 1
    if y_train[i] == 5:
        y_train5[i] = 1

N_train = len(y_train)
N_test = len(y_test)

y_test1 = np.negative(np.ones((N_test))).astype(int)
y_test2 = np.negative(np.ones((N_test))).astype(int)
y_test3 = np.negative(np.ones((N_test))).astype(int)
y_test4 = np.negative(np.ones((N_test))).astype(int)
y_test5 = np.negative(np.ones((N_test))).astype(int)

for i in range(N_test):
    if y_test[i] == 1:
        y_test1[i] = 1
    elif y_test[i] == 2:
        y_test2[i] = 1
    elif y_test[i] == 3:
        y_test3[i] = 1
    elif y_test[i] == 4:
        y_test4[i] = 1
    elif y_test[i] == 5:
        y_test5[i] = 1

        # define Gaussian kernel function

#This is different from Gaussian density, this is not normalized
#The integration will not result in 1. there is no 1/2pi coefficient here.
def gaussian_kernel(X1, X2, s): #same for all classes

    D = dt.cdist(X1, X2)
    K = np.exp(-D**2 / (2 * s**2))
    return(K)

def yyk_calculator(y, kernel):
    yyK = np.matmul(y[:,None], y[None,:]) * kernel
                    #(1000, 1)   #(1, 1000)
    return yyK

def get_num_sample(y):
    # get number of samples and number of features
    N_set = len(y)
    return N_set

def train_param(C, x, y):
    # calculate Gaussian kernel
    s = 10 #width parameter of 1
    kernel = gaussian_kernel(x, x, s)
    yyK = yyk_calculator(y, kernel)
    N_set = get_num_sample(y)

    epsilon = 1e-3

    #we are doing optimization with respect to alpha.
    P = cvx.matrix(yyK) #alphas are quadratic. #this is a quadratic programming.
    q = cvx.matrix(-np.ones((N_set, 1)))
    G = cvx.matrix(np.vstack((-np.eye(N_set), np.eye(N_set))))
    h = cvx.matrix(np.vstack((np.zeros((N_set, 1)), C * np.ones((N_set, 1)))))
    A = cvx.matrix(1.0 * y[None,:])
    b = cvx.matrix(0.0)

    # use cvxopt library to solve QP problems
    result = cvx.solvers.qp(P, q, G, h, A, b)
    alpha = np.reshape(result["x"], N_set)
    alpha[alpha < C * epsilon] = 0
    alpha[alpha > C * (1 - epsilon)] = C

    # find bias parameter
    support_indices, = np.where(alpha != 0) #the data points with non zero alpha coefficients are support indecies
    active_indices, = np.where(np.logical_and(alpha != 0, alpha < C))
    w0 = np.mean(y[active_indices] * (1 - np.matmul(yyK[np.ix_(active_indices, support_indices)], alpha[support_indices])))
    return [w0, alpha]

y_train_list = [y_train1, y_train2, y_train3, y_train4, y_train5]
C_list = [0.1,1, 10, 100, 1000]

all_params = []
for y in y_train_list:
    current_param = []
    for c in C_list:
        current_param.append(train_param(c, x_train, y))
    all_params.append(current_param)
def cal_prediction(x, y, w0, alpha):
    s=10
    kernel = gaussian_kernel(x, x_train, s)
    f_predicted = np.matmul(kernel, y[:,None] * alpha[:,None]) + w0
    return f_predicted

#calculate maximum for c = 10
def prediction_array(arr, N):
    y_prediction = np.zeros(N).astype(int)
    y_prediction = np.argmax(arr, axis=0) + 1
    return y_prediction
#[i][j][k] i: class number j: C idx K: 0=w0 and 1=alpha
predictions_c_10 = []

for i in range(K):
    result = cal_prediction(x_train, y_train_list[i], all_params[i][2][0], all_params[i][2][1])
    predictions_c_10.append(result)
prediction = prediction_array(predictions_c_10, 1000).reshape(-1)

confusion_matrix = pd.crosstab(np.reshape(prediction, N_train), y_train, rownames = ['y_predicted'], colnames = ['y_train'])
print(confusion_matrix)
print("Calculating Confusion matrix for test data ...")
#[i][j][k] i: class number j: C idx K: 0=w0 and 1=alpha
predictions_c_10 = []
y_test_list = [y_test1, y_test2, y_test3, y_test4, y_test5]
for i in range(K):
    result = cal_prediction(x_test, y_train_list[i], all_params[i][2][0], all_params[i][2][1])
    predictions_c_10.append(result)
prediction = prediction_array(predictions_c_10, 4000).reshape(-1)

# calculate confusion matrix
confusion_matrix = pd.crosstab(prediction, y_test, rownames = ['y_predicted'], colnames = ['y_test'])
print(confusion_matrix)
def prediction_accuracy(C, x, y, N, param):
    idx = C_list.index(C)
    p=[]
    for i in range(K):
        result = cal_prediction(x, y_train_list[i], param[i][idx][0], param[i][idx][1])
        p.append(result)
    prediction = prediction_array(p, N).reshape(-1)
    return np.mean(prediction == y)

C_train_accuracies = []
C_test_accuracies = []
print("Plotting prediction accuracies for different C values. Please wait...")
for c in C_list:
    temp = []
    temp = prediction_accuracy(c, x_train, y_train, 1000, all_params)
    C_train_accuracies.append(temp)

for c in C_list:
    temp = []
    temp = prediction_accuracy(c, x_test, y_test, 4000, all_params)
    C_test_accuracies.append(temp)

x_coordinates = C_list
x = np.stack((x_coordinates, C_train_accuracies))
x2 = np.stack((x_coordinates, C_test_accuracies))
y_ticks = [0, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
plt.figure(figsize = (8, 6))
plt.axes()
plt.xscale("log")

plt.yticks(ticks = y_ticks)
plt.xlabel("x1")
plt.ylabel("x2")
plt.plot(x[0], x[1],"-o", color = "blue", markersize = 8, label = "training")
plt.plot(x2[0], x2[1], "-o", color = "red",markersize = 8, label = "test")
plt.legend()
plt.show()
