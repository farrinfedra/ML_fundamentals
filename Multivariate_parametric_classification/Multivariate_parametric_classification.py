import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from numpy.linalg import inv
import pandas as pd

##Parameters
np.random.seed(421)
class_means = np.array([[0.0, 2.5], [-2.5, -2.0], [2.5, -2.0]])
class1_covariance = np.matrix([[3.2, 0], [0, 1.2]])
class2_covariance = np.matrix([[1.2, 0.8], [0.8, 1.2]])
class3_covariance = np.matrix([[1.2, -0.8], [-0.8, 1.2]])

class_sizes = np.array([120, 80, 100])

#Data Generation
# generate random samples
points1 = np.random.multivariate_normal(class_means[0], class1_covariance, class_sizes[0])
points2 = np.random.multivariate_normal(class_means[1], class2_covariance, class_sizes[1])
points3 = np.random.multivariate_normal(class_means[2], class3_covariance, class_sizes[2])

points = np.concatenate((points1, points2, points3))
# generate corresponding labels
y = np.concatenate((np.repeat(1, class_sizes[0]), np.repeat(2, class_sizes[1]),
                                                   np.repeat(3, class_sizes[2])))

# Parameter Estimation
# get number of classes and number of samples
K = np.max(y)
N = points.shape[0]
D = points.shape[1]
sample_means = np.array([np.mean(points[y == c + 1], axis = 0).reshape(2,1) for c in range(K)])
print(sample_means)

#covariances
sample_covariance1 = np.cov(points[y == 1] , rowvar = False)
sample_covariance2 = np.cov(points[y == 2] , rowvar = False)
sample_covariance3 = np.cov(points[y == 3] , rowvar = False)
sample_covariances = np.array([sample_covariance1, sample_covariance2, sample_covariance3])
print(sample_covariances)
# prior probabilities
class_priors = [np.mean(y == (c + 1)) for c in range(K)]
print(class_priors)

def getScoreOfClass(class_mean, class_covariance, class_prior, x):
    covariance_inverse = np.linalg.inv(class_covariance)
    W_c = (-0.5) * covariance_inverse
    w_c = np.matmul(class_mean, covariance_inverse)
    A = np.matmul(np.transpose(class_mean), covariance_inverse)
    det = np.linalg.det(class_covariance)

    w_c0 = np.matmul(A, class_mean) - (0.5 * np.log(det)) + np.log(class_prior)

    B = np.dot(np.transpose(x), W_c)

    score = np.dot(B, x) + np.dot(np.transpose(w_c), x) + w_c0

    return score


def getScores(x):
    scores = np.array([getScoreOfClass(class_means[i], sample_covariances[i], class_priors[i], x) for i in range(K)])
    return np.argmax(scores) + 1

estimates = [getScores(x) for x in points]

y = y.flatten()
estimates = np.array(estimates)
confusion_matrix = pd.crosstab(estimates, y, rownames = ['y_pred'], colnames = ['y_truth'])
print(confusion_matrix)

#plotting
plt.figure(figsize = (8, 8))
plt.plot(points[y == 1, 0], points[y == 1, 1], "r.", markersize = 10)
plt.plot(points[y == 2, 0], points[y == 2, 1], "g.", markersize = 10)
plt.plot(points[y == 3, 0], points[y == 3, 1], "b.", markersize = 10)
plt.xlabel("x1")
plt.ylabel("x2")
plt.plot(points[y != estimates, 0], points[y != estimates, 1], "ko", markersize = 12, fillstyle = "none")
plt.show()
