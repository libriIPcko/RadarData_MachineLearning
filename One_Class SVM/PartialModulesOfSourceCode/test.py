import numpy as np

array_3d = np.empty((1,3))

print(array_3d)

np.savetxt('test.csv',array_3d,delimiter=',')