import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.linear_model import SGDOneClassSVM
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline

import cv2
import time


random_state = 40
rng = np.random.RandomState(random_state)
# Generate train data
X = 0.3 * rng.randn(500, 2)
data_train = np.r_[X + 2, X - 2]

font = {"weight": "normal", "size": 15}
matplotlib.rc("font", **font)

#path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/ParsedData/parsOut_18.4__11_32_13_xwr18xx_processed_stream_2023_03_17T12_02_36_082.csv'
#path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/ParsedData/parsOut_18.4__11_32_25_static_xwr18xx_processed_stream.csv'
path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/ParsedData/parsOut_18.4__11_39_3_static_v2_xwr18xx_processed_stream.csv'
#path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/ParsedData/parsOut_18.4__11_39_39_static_v1_xwr18xx_processed_stream.csv'
#path = 'C:/Users/bob/Documents/GitHub/RadarData_MachineLearning/RadarData_MachineLearning/ParsedData/parsOut_18.4__11_40_7_dynamic_xwr18xx_processed_stream.csv'
data = np.genfromtxt(path,delimiter=',',skip_header=1)
data_all = np.genfromtxt(path,delimiter=',',skip_header=1)
focusedFrame = 0
# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
vid_fileName = "SVMmethod"+path.split('/')[-1].split('.')[-2]+".mp4"
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(0, 6, 50))
video_writer = cv2.VideoWriter(vid_fileName, fourcc, 30, (640, 480))

startTime_total = time.process_time()
i=0
#lastFrame = data_all[-1,0]
lastFrame = 10
while(i<lastFrame):
    startTime = time.process_time()
    focusedFrame = i
    # selection data_train of focusedFrame data +1
    # selection on focusedFrame data
    #indices_train = np.argwhere(data_all[:, 0] == focusedFrame)
    #indices_train = np.squeeze(indices_train)
    #data_train = np.hstack((np.array(data_all[indices_train[0]:indices_train[-1] + 1, 2]).reshape(-1, 1),np.array(data_all[indices_train[0]:indices_train[-1] + 1, 3]).reshape(-1, 1)))
    # selection on focusedFrame data
    indices = np.argwhere(data_all[:, 0] == focusedFrame)
    indices = np.squeeze(indices)
    data = np.hstack((np.array(data_all[indices[0]:indices[-1] + 1, 2]).reshape(-1, 1), np.array(data_all[indices[0]:indices[-1] + 1, 3]).reshape(-1, 1)))
    #plot configuration
    plt.clf()
    plt.figure(1)
    plt.xlim(-5, 5)
    plt.ylim(0, 6)
    plt.xlabel(
        "error train: %d/%d; errors novel regular: %d/%d; errors novel abnormal: %d/%d"
        % (
            n_error_train,
            data_train.shape[0],
            n_error_test,
            data.shape[0],
        )
    )

    # OCSVM hyperparameters
    nu = 0.05
    gamma = 2.0
    # Fit the One-Class SVM
    clf = OneClassSVM(gamma=gamma, kernel="rbf", nu=nu)
    clf.fit(data_train)
    y_pred_train = clf.predict(data_train)
    n_error_train = y_pred_train[y_pred_train == -1].size
    y_pred_test = clf.predict(data)
    n_error_test = y_pred_test[y_pred_test == -1].size
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Fit the One-Class SVM using a kernel approximation and SGD
    transform = Nystroem(gamma=gamma, random_state=random_state)
    clf_sgd = SGDOneClassSVM(
        nu=nu, shuffle=True, fit_intercept=True, random_state=random_state, tol=1e-4
    )
    pipe_sgd = make_pipeline(transform, clf_sgd)
    pipe_sgd.fit(data_train)
    y_pred_train_sgd = pipe_sgd.predict(data_train)
    y_pred_test_sgd = pipe_sgd.predict(data)
    n_error_test = y_pred_test_sgd[y_pred_test_sgd == -1].size
    Z_sgd = pipe_sgd.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z_sgd = Z_sgd.reshape(xx.shape)
    # plot the level sets of the decision function
    s = 20
    plt.figure(figsize=(9, 6))
    plt.title("One Class SVM")
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
    a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors="darkred")
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors="palevioletred")
    b1 = plt.scatter(data_train[:, 0], data_train[:, 1], c="white", s=s, edgecolors="k")
    b2 = plt.scatter(data[:, 0], data[:, 1], c="blueviolet", s=s, edgecolors="k")
    plt.axis("tight")
    plt.xlim((-5, 5))
    plt.ylim((0, 6))

    # save the plot as an image
    fig = plt.gcf()
    #To vid
    # Save the figure as a PNG image
    fig.savefig(f"figure_radDat_mean_shift/frame_{i:04d}.png")
    # Load the image and write it to the video file
    img = cv2.imread(f"figure_radDat_mean_shift/frame_{i:04d}.png")
    video_writer.write(img)
    endTime = time.process_time()
    print("Current progress: %d/%d - Duration: %.5f" %(focusedFrame,lastFrame,(endTime-startTime)))
    i=i+1
video_writer.release()
endTime_total = time.process_time()
print("Total time: %.5f" %(endTime_total-startTime_total))