import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
#from sklearn.datasets import make_blob
import matplotlib.pyplot as plt

path_1440frames = 'C:/Users/bob\Documents/build-RadarVisualizer-Desktop_Qt_6_4_2_MinGW_64_bit-Release/release/parse_script/ParsedData/parsOut_5.4__22_12_5.csv'
path_199frames = 'C:/Users/bob\Documents/build-RadarVisualizer-Desktop_Qt_6_4_2_MinGW_64_bit-Release/release/parse_script/ParsedData/parsOut_16.3__19_5_43.csv'
data = np.genfromtxt(path_199frames,delimiter=',',skip_header=1)
focusedFrame = 151

#selection on focusedFrame data
indices = np.argwhere(data[:,0] == focusedFrame)
indices = np.squeeze(indices)
data_posX_focusedFrame = np.array(data[indices[0]:indices[-1]+1,2])
data_posY_focusedFrame = np.array(data[indices[0]:indices[-1]+1,3])
x = data_posX_focusedFrame
y = data_posY_focusedFrame
data = np.hstack((x.reshape(-1,1),y.reshape(-1,1)))
#print(data)

#implementation mean-shift algorithm
# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(data, quantile=0.2)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(data)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

plt.figure(2)
plt.clf()

colors = ["#dede00", "#377eb8", "#f781bf"]
markers = ["x", "o", "^"]

for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(data[my_members, 0], data[my_members, 1], markers[k], color=col)
    plt.plot(
        cluster_center[0],
        cluster_center[1],
        markers[k],
        markerfacecolor=col,
        markeredgecolor="k",
        markersize=14,
    )
plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.show()


#fig, ax = plt.subplots()
#ax.scatter(x,y)
#ax.set(xlim=(x[0], x[-1]), xticks=np.arange(0, x[-1]), ylim=(np.min(y), np.max(y)), yticks=np.arange(0, np.max(y)))
#plt.show()



#outarray = np.column_stack((x,y))
#print (outarray)
#output_path = 'C:/Users/bob/PycharmProjects/pythonProject2'
#np.savetxt('feedbackData.csv',outarray,delimiter=',',fmt='%.4f')