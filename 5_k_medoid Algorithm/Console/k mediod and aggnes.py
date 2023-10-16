#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries and loading dataset
import numpy as np
#from sklearn import datasets
#df=datasets.load_iris().data

import pandas as pd
#pd.set_option('display.max_columns', None)
df=pd.read_csv('C:/Users/Sara Maher/Mall_Customers.csv', sep=',')
#df=pd.read_csv('C:/Users/Sara Maher/Mall_Customers.csv', delimiter=',')
df.head()


# In[2]:


df.shape


# In[3]:


df= df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
df=df.values
print(df)
#df.head()


# In[4]:


df.shape


# In[5]:


import random
import copy


# In[6]:


def euclidean_distance(a, b):          #euclidean distance between a and b, datapoints
    z=np.sqrt(sum(np.square(a-b)))
    return z


def point_distribution(matrix,centroid_list):

    cluster_points = [[i] for i in centroid_list]
    label_list = []
    for datapoint in matrix:
        # Calculate the distance from the node point to each center and divide it to the nearest center point
        dist_list = [euclidean_distance(datapoint, centroid) for centroid in centroid_list]
        label = np.argmin(dist_list)  # Select the nearest cluster center
        label_list.append(label)
        cluster_points[label].append(datapoint)  # Add point to the nearest cluster
    return label_list, cluster_points


# In[7]:


def pam(data, k):

    # Random initial cluster center
    index_list = list(range(len(data)))
    random.shuffle(index_list)
    shuffled_index = index_list[:k]
    centroids = data[shuffled_index, :]  # Array of center points
    labels = []  # Category label for each data
    stop_flag = False  # A sign that the algorithm stops iterating
    while not stop_flag:
        stop_flag = True
        cluster_points = [[i] for i in centroids]  # The i-th element is a collection of data points of the i-th type
        labels = []  # Category label for each data
        #Iterate over the data
        for datapoint in data:
            #Calculate the distance from the node point to each center and divide it to the nearest center point
            distances = [euclidean_distance(datapoint, i) for i in centroids]
            label = np.argmin(distances)  # Select the nearest cluster center
            labels.append(label)
            cluster_points[label].append(datapoint)  #Add point to the nearest cluster

        #Calculate the total distance between the current center point and all other points
        distances = []
        for i in range(k):
            distances.extend([euclidean_distance(j, centroids[i]) for j in cluster_points[i]])
        old_distances_sum = sum(distances)

        #Try to replace the center point with each non-central point in the entire data set. If the clustering error is reduced, change the center point
        for i in range(k):
            # Calculate the distance from each node to the center of the original cluster in the i-th cluster
            for datapoint in data:
                new_centroids = copy.deepcopy(centroids)  #Hypothetical center set
                new_centroids[i] = datapoint
                labels, cluster_points = point_distribution(data, new_centroids)
                #Calculate new clustering error
                distances = []
                for j in range(k):
                    distances.extend([euclidean_distance(p, new_centroids[j]) for p in cluster_points[j]])
                new_distances_sum = sum(distances)

                #Determine whether the clustering error is reduced
                if new_distances_sum < old_distances_sum:
                    old_distances_sum = new_distances_sum
                    centroids[i] = datapoint  #Modify the center of the i-th cluster
                    stop_flag = False
    return centroids, labels, old_distances_sum


# In[8]:


def k_medoid_clustering(data,k):
    centroids,targets,old_dis=pam(data,k)
    print("CENTROIDS: \n",centroids)
    print("TARGETS:",targets)
    #print("old_dis:",old_dis)


# In[9]:


k_medoid_clustering(df,3)


# AgglomerativeClustering

# In[11]:


import matplotlib.pyplot as plt  
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(df[:,:], method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()


# In[12]:


def euclidean_dist(a, b):          #euclidean distance between a and b, datapoints
    return np.sqrt(sum(np.square(a-b)))
#TAKES IN TWO CLUSTERS. RETURNS SINGLE_LINK DISTANCE OR MINIMUM DISTANCE BETWEEN THEM. 
def single_link_dist(cluster1,cluster2):       
    dist_list=list()
    for i in cluster1:                  #datapoint(s) of cluster1
        for j in cluster2:              #datapoint(s) of cluster2
            dist_list.append(euclidean_dist(i,j))
    return(min(dist_list))


# In[13]:


def aglomerative(data,k):
    """parameter data: data array
       parameter k: Number of clusters 
    """
    N=len(data)
    cluster_label=[[i] for i in range(N)]
    
    if k == N:
        return cluster_label
    else:
        for cluster_num in range(N-1,k-1,-1):
            
            counter=0
            for i in range(len(cluster_label)-1):
                cluster1=list()
                for t in cluster_label[i]:
                    cluster1.append(data[t])
                
                for j in range(i+1,len(cluster_label)):
                    cluster2=list()
                    for t1 in cluster_label[j]:
                        cluster2.append(data[t1])
                    
                    if counter == 0:
                        min_sl_dist=single_link_dist(cluster1,cluster2)
                        r,c=0,1
                        counter=2
                    else:
                        if single_link_dist(cluster1,cluster2)<min_sl_dist:
                            min_sl_dist = single_link_dist(cluster1,cluster2)
                            r,c=i,j
            
            cluster_label[r]=cluster_label[r]+cluster_label[c]
            del cluster_label[c]
    return cluster_label
    


# In[14]:


clus=aglomerative(df,10)


# In[15]:


for i in range(len(clus)):
    print("CLUSTER",i+1,":\n",clus[i])
print(len(clus))

