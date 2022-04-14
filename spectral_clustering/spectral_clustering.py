import numpy as np
from matplotlib import pyplot as plt
from scipy import spatial as spa
import scipy.linalg as linalg

def update_centroids(memberships, Z):
    if memberships is None:
        # initialize centroids
        centroids = Z[(29, 143, 204, 271, 277),:]
    else:
        # update centroids
        centroids = np.vstack([np.mean(Z[memberships == k,], axis = 0) for k in range(K)])
    return centroids

def update_memberships(centroids, Z):
    # calculate distances between centroids and data points
    D = spa.distance_matrix(centroids, Z)
    # find the nearest centroid for each data point
    memberships = np.argmin(D, axis = 0)
    return memberships

X = np.genfromtxt("hw08_data_set.csv", delimiter = ",")
N = X.shape[0]
threshold = 1.25

distances = spa.distance_matrix(X, X)

B = np.zeros((N, N))
B[distances <= threshold] = 1
np.fill_diagonal(B, 0)

plt.figure(figsize = (12,12))
plt.gca().set_aspect('equal', adjustable='box')
plt.plot(X[:,0], X[:,1], "ko")
for i in range(N):
    for j in np.nonzero(B[i])[0]:
        plt.plot([X[i,0], X[j,0]], [X[i,1], X[j,1]], "k", lw = 0.5)

plt.show()


D = np.zeros((N, N))
for i in range(N):
    D[i][i] = sum(B[i] > 0)

L = D - B

for i in range(D.shape[0]):
    D[i][i] = D[i][i]**(-1/2)

laplacian = np.eye(N) - np.matmul(np.matmul(D, B), D)

values, vectors = linalg.eig(laplacian)
values = np.real(values)
vectors = np.real(vectors)

R = 5
Z = vectors[:,values.argsort()[0:R]]


K = 5
centroids = None
memberships = None
iteration = 1
while True:
    print("iteration: ",iteration)
    old_centroids = centroids
    centroids = update_centroids(memberships, Z)
    if np.alltrue(centroids == old_centroids):
        break

    old_memberships = memberships
    memberships = update_memberships(centroids, Z)
    if np.alltrue(memberships == old_memberships):
        break
    iteration += 1

plt.figure(figsize=(12,12))
cluster_colors = np.array(["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"])
for c in range(K):
    plt.plot(X[memberships == c, 0], X[memberships == c, 1], "o", markersize = 10,
             color = cluster_colors[c])

plt.show()

