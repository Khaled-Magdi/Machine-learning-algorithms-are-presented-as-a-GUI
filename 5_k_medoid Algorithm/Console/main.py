import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# preparing dataset

dataset = pd.read_csv("Mall_Customers.csv")

#############################################################################

# take only 3 features
data1 = dataset[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
dataValues = data1.values

# visualising the data before clustring
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(data1["Age"], data1["Annual Income (k$)"], data1["Spending Score (1-100)"], c='purple', s=60)
plt.title("simple 3D scatter plot")
ax.view_init(35, 185)
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
ax.set_zlabel('Spending Score (1-100)')
plt.show()

# 1) the initial medoids
k = 3


def medInit(data, k):
    import random
    meds = random.sample(range(0, len(data) - 1), k)  # 0-199 3 numbers [18,50,0] (indices)
    return data[meds, :]  # data[0,:] (values)


old_medoids = medInit(dataValues, 3)

# 2) distance function
from scipy.spatial import distance


def computeDistance(data, medoids):
    m = len(data)  # 200
    medoids_shape = medoids.shape
    # If a 1-D array is provided,
    # it will be reshaped to a single row 2-D array
    if len(medoids_shape) == 1:
        medoids = medoids.reshape((1, len(medoids)))
    j = len(medoids)  # 3
    disMatrix = np.empty((m, j))  # 200x3
    for i in range(m):  # 0-199 , # i =1
        arr = []
        for n in range(j):  # 0-2 #n =0  dataValues[0,:] [50 7 2]
            arr.append(distance.euclidean(dataValues[i, :], medoids[n, :]))  # (x,y,z) - (x1,y1,z1)
        disMatrix[i, :] = arr  # i 1
    return disMatrix


dis = computeDistance(dataValues, old_medoids)


# np.sum(dis)
def assign_labels(distance):
    return np.argmin(distance, axis=1)


labels = assign_labels(dis)


# data1["labels"] = labels
# cl1 = data1[labels==0]

# update medoids
def update(data, medoids):
    dis = computeDistance(data, medoids)
    labels = assign_labels(dis)
    out_medoids = medoids

    for i in set(labels):  # i = 0 ,1 ,2 (k times)
        avgDissimilarity = np.sum(computeDistance(data, medoids))

        clusters_points = data[labels == i]  # len(data1[labels ==0])
        for dataPoint in clusters_points:  #
            new_medoid = dataPoint
            newDissimilarity = np.sum(computeDistance(data, new_medoid))  # new_medoid
            if newDissimilarity < avgDissimilarity:
                avgDissimilarity = newDissimilarity
                out_medoids[i] = dataPoint
    return out_medoids


new_medoids = update(dataValues, old_medoids)


# function to check whether the medoids no longer move and the iteration should be stopped.
def has_converged(old_medoids, medoids):
    return set([tuple(x) for x in old_medoids]) == set([tuple(x) for x in medoids])


# lao rg3 true he3ml stop


####################################################################
#################### FULL ALGORITHM ##########################
def k_medoid(data, k, starting_medoids=None, max_steps=np.inf):
    if starting_medoids is None:
        medoids = medInit(data, k)
    else:
        medoids = starting_medoids

    converged = False
    labels = np.zeros(len(data))  # 200
    i = 1
    while (not converged) and (i <= max_steps):
        old_medoids = medoids.copy()

        dis = computeDistance(data, medoids)
        labels = assign_labels(dis)
        medoids = update(data, medoids)
        converged = has_converged(old_medoids, medoids)
        i += 1

    return (medoids, labels)


medoids, labels = k_medoid(dataValues, 3, None)

# adding labels into the dataset to draw ot
data1["label"] = labels

# visualizing after clustring
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data1.Age[data1.label == 0], data1["Annual Income (k$)"][data1.label == 0],
           data1["Spending Score (1-100)"][data1.label == 0], c='purple', s=60)
ax.scatter(data1.Age[data1.label == 1], data1["Annual Income (k$)"][data1.label == 1],
           data1["Spending Score (1-100)"][data1.label == 1], c='red', s=60)
ax.scatter(data1.Age[data1.label == 2], data1["Annual Income (k$)"][data1.label == 2],
           data1["Spending Score (1-100)"][data1.label == 2], c='blue', s=60)
# ax.scatter(data1.Age[data1.label == 3], data1["Annual Income (k$)"][data1.label == 3], data1["Spending Score (1-100)"][data1.label == 3], c='green', s=60)
# ax.scatter(data1.Age[data1.label == 4], data1["Annual Income (k$)"][data1.label == 4], data1["Spending Score (1-100)"][data1.label == 4], c='yellow', s=60)
ax.view_init(35, 185)
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
ax.set_zlabel('Spending Score (1-100)')
plt.show()