import numpy as np
import matplotlib.pyplot as plt


path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/Datasets/static_measurement_parsed/mer2.csv'
data = np.genfromtxt(path,delimiter=',',skip_header=1)
data_all = np.genfromtxt(path,delimiter=',',skip_header=1)
fileName = path[path.rfind('/') + 1:]

radius = 1
pos_x = 2.5
pos_y = 8
pos_y_min = pos_y - 1
pos_y_max = pos_y + 1
pos_x_min = pos_x - 1
pos_x_max = pos_x + 1

mirror = False

if mirror == True:
    pos_x = pos_x * -1

#lastFrame = data_all[-1,0]
lastFrame = 2
i=0
while(i < lastFrame):
    focusedFrame = i
    indices = np.argwhere(data_all[:, 0] == focusedFrame)
    indices = np.squeeze(indices)
    procData = data[indices[0]:indices[-1],:]
    np.savetxt('test1.csv', procData, delimiter=',')
    #plt.clf()
    #plt.scatter(procData[indices[0]:indices[-1],2],procData[indices[0]:indices[-1],3])
    #plt.show()

    j = 0
    while(j<(indices[-1]-indices[0])):
        # Define column
        objColumn = np.empty((indices[-1] - indices[0], 1))
        data_x = procData[j, 2]
        data_y = procData[j, 3]
        # comparision in x:
        if data_x < pos_x_max and data_x > pos_x_min:
            # comparision in y:
            if data_y < pos_y_max and data_y > pos_y_min:
                objColumn[j,0] = True
                #print('ok')
            else:
                objColumn[j,0] = False
        else:
            objColumn[j,0] = False
        j = j + 1
        procData.shape[0]
        objColumn.shape[1]
        #print(objColumn)
        procData = np.hstack((procData,objColumn))
        #print(procData)

    #print(procData)
    np.savetxt('test.csv',procData,delimiter=',')
    i = i + 1
