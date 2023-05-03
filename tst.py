import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])
kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(X)
kmeans.labels_

kmeans.predict([[0, 0], [12, 3]])


fig = plt.figure()
ax1 = plt.subplot(211)
ax1.set_title("raw values")
ax1 = plt.scatter(X[:, 0],X[:, 1], c='green')

ax2 = plt.subplot(212)
ax2.set_title("cluster centers of predicted values")
ax2 = plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c='red')
plt.subplots_adjust(hspace=0.5)
plt.show()