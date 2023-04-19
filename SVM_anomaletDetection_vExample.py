# import libraries
import pandas as pd
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import numpy as np
from numpy import where



# import data
data = pd.read_csv("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv")# input data
df = data[["sepal_length", "sepal_width"]]
# model specification
model = OneClassSVM(kernel = 'rbf', gamma = 0.001, nu = 0.03).fit(df)
# prediction
y_pred = model.predict(df)
#y_pred
# filter outlier index
outlier_index = where(y_pred == -1) # filter outlier values
outlier_values = df.iloc[outlier_index]
outlier_values
# visualize outputs
plt.scatter(data["sepal_length"], df["sepal_width"])
plt.scatter(outlier_values["sepal_length"], outlier_values["sepal_width"], c = "r")
plt.show()
print("Det anomaly: %d" %np.array(outlier_values).shape[1])