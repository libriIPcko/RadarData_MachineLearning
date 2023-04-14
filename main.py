import numpy as np
from sklearn import kernel_approximation

import matplotlib.pyplot as plt



rng = np.random.RandomState(0)
X = rng.rand(10, 2000)
X = np.array(X, dtype='float32')
X.dtype
#dtype('float32')
print(X)

transformer = kernel_approximation.RBFSampler()
X_new = transformer.fit_transform(X)
X_new.dtype
#dtype('float32')

print('after transformation')
print(X_new)

plt.style.use('_mpl-gallery')
# focused array column or row
y = X[1,:]
x = np.arange(len(y))
# size and color:
sizes = np.random.uniform(15, 80, len(y))
colors = np.random.uniform(15, 80, len(y))


fig, ax = plt.subplots()
#ax.scatter(x,y,s=sizes, c=colors, vmin=0, vmax=100)
ax.scatter(x,y)
ax.set(xlim=(x[0], x[-1]), xticks=np.arange(0, x[-1]),
       ylim=(np.min(y), np.max(y)), yticks=np.arange(0, np.max(y)))

plt.show()
