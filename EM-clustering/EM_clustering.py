import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
import scipy.spatial as spa

X = np.genfromtxt("hw07_data_set.csv", delimiter = ",")
initial_centroids = np.genfromtxt("hw07_initial_centroids.csv", delimiter = ",")


N = X.shape[0]
D = X.shape[1]
K = initial_centroids.shape[0]


means = np.array([[+2.5, +2.5],
                        [-2.5, +2.5],
                        [-2.5, -2.5],
                        [+2.5, -2.5],
                        [+0.0, +0.0]])
covs = np.array([[[+0.8, -0.6],
                               [-0.6, +0.8]],
                              [[+0.8, +0.6],
                               [+0.6, +0.8]],
                              [[+0.8, -0.6],
                               [-0.6, +0.8]],
                              [[+0.8, +0.6],
                               [+0.6, +0.8]],
                              [[+1.6, +0.0],
                               [+0.0, +1.6]]])

centroids = initial_centroids
memberships = np.argmin(spa.distance_matrix(centroids, X), axis = 0)
covariance = np.asarray([np.cov(X[memberships == c].T) for c in range(K)])

initial_covs = covariance

priors = np.asarray([np.mean(memberships == c) for c in range(K)])

H = np.zeros((N,K))

def calculate_priors(H):
    return(np.mean(H, axis = 0))

def calculate_centroids(X, H):
    cs =[]
    for c in range(K):
        ml = np.matmul(H[c], X)
        ml = ml / H[c, :].sum()
        cs.append(ml)
    return np.asarray(cs)


def calculate_covariance(X, H, centroids):
    covariance = np.zeros((K, D, D))
    for c in range(K):
        for i in range(N):
            covariance[c] += H[i][c]*np.matmul((X[i] - centroids[c]).reshape(2,1), (X[i] - centroids[c]).reshape(1,2))
        covariance[c] /= H[:,c].sum()
    return covariance

    


def update_memberships(centroids, X):
    # calculate distances between centroids and data points
    D = spa.distance_matrix(centroids, X)
    # find the nearest centroid for each data point
    memberships = np.argmin(D, axis = 0)
    return memberships




iteration = 100
for i in range(iteration):

    #E-step

    H_nom = np.asarray([stats.multivariate_normal.pdf(X, mean = centroids[c], cov = covariance[c]) * priors[c] for c in range(K)]).T
    H = np.asarray([H_nom[:,c] / np.sum(H_nom,axis=1) for c in range(K)]).T

    centroids = calculate_centroids(X, H.T)
    priors = calculate_priors(H)
    covariance = calculate_covariance(X, H, centroids)
    memberships = update_memberships(centroids, X)


print(centroids)
colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928","#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])

plt.figure(figsize=(8, 8))
for c in range(K):
    plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize = 10,
                     color = colors[c])

x1_interval = np.arange(-5, +5, 0.05)
x2_interval = np.arange(-5, +5, 0.05)
x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
coordinates = np.empty(x1_grid.shape + (2,))
coordinates[:, :, 0] = x1_grid
coordinates[:, :, 1] = x2_grid

for i in range(K):
    pdf1 = stats.multivariate_normal(means[i], covs[i])
    pdf2 = stats.multivariate_normal(centroids[i], covariance[i])
    plt.contour(x1_grid, x2_grid, pdf1.pdf(coordinates), colors = 'k',linestyles='dashed', levels=[0.05])
    plt.contour(x1_grid, x2_grid, pdf2.pdf(coordinates),colors =colors[i], levels=[0.05])

plt.show()
