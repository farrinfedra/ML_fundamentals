import matplotlib.pyplot as plt
import numpy as np
import math
from pandas import crosstab

def safelog(x):
    return np.log(x + 1e-100)

data_set_images = np.genfromtxt("hw02_images.csv", delimiter = ",")
data_set_labels = np.genfromtxt("hw02_labels.csv", delimiter = "\n")

# training dataset
train_set_x = data_set_images[0:30000,:]
train_set_y = data_set_labels[0:30000].astype(int)

#testing dataset
test_set_x = data_set_images[30000:35000, :]
test_set_y = data_set_labels[30000:35000].astype(int)


N = train_set_x.shape[0]
D = train_set_x.shape[1]
K = np.max(train_set_y)

sample_means = np.array((np.mean(train_set_x[train_set_y == 1], axis = 0),
                            np.mean(train_set_x[train_set_y == 2], axis = 0),
                            np.mean(train_set_x[train_set_y == 3], axis = 0),
                            np.mean(train_set_x[train_set_y == 4], axis = 0),
                            np.mean(train_set_x[train_set_y == 5], axis = 0))).T
print(sample_means)

sample_deviations = np.array((
                        np.std(train_set_x[train_set_y == 1], axis = 0),
                        np.std(train_set_x[train_set_y == 2], axis = 0),
                        np.std(train_set_x[train_set_y == 3], axis = 0),
                        np.std(train_set_x[train_set_y == 4], axis = 0),
                        np.std(train_set_x[train_set_y == 5], axis = 0))).T
print(sample_deviations)

#class priors
class_priors = np.array([np.mean(train_set_y == (c + 1)) for c in range(K)])
print(class_priors)

def score_function(x):
    constants = -D/2 * np.log(2 * math.pi)
    total = [
            - np.sum(safelog(sample_deviations[:, c]))
            - np.sum(((x - sample_means[:, c]) ** 2) / (2 * (sample_deviations[:, c] ** 2)))
            + np.log(class_priors[c]) + constants
        for c in range(K)

    ]
    return total

train_scores_y = [score_function(x) for x in train_set_x]
pred_y = np.argmax(train_scores_y, axis = 1) + 1

print(crosstab(pred_y, train_set_y, rownames=["y_pred"], colnames=["y_truth"]))

#score function on test dataset
test_scores_y = [score_function(x) for x in test_set_x]
pred_test_y = np.argmax(test_scores_y, axis = 1) + 1

print(crosstab(pred_test_y, test_set_y, rownames=["y_pred"], colnames=["y_truth"]))
