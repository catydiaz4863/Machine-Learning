from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np

df = np.loadtxt("/Users/catalinadiaz/Documents/ML_fall2021/auto-mpg_removed_missing_values.data", usecols=(0,1,2,3,4,5,6,7))


#Normalization
scaler = MinMaxScaler()
scaler.fit(df[:, 1:])
df[:, 1:] = scaler.transform(df[:, 1:])

##Multivariable Linear Regression
y = df[:, 0]
x = df[:, 1:]


#initialization
w = np.array([0, 0, 0, 0, 0, 0, 0])
b = 0

#learning rate
alpha = 0.01

#gd

for i in range(50000):
    w = w - alpha * (1/ len(df)) * np.dot(np.transpose(np.dot(x,w)+b - y), x)
    b = b - alpha * (1/ len(df)) * sum(np.dot(x, w)+b - y)

print(w,b)

