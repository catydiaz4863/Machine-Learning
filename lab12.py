
import perform as perform
import splits as splits
from matplotlib.pyplot import clf
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn.utils import shuffle
import pandas as pd

df = np.loadtxt("/Users/catalinadiaz/Documents/ML_fall2021/auto-mpg_removed_missing_values.data", usecols=(0,1,2,3,4,5,6,7))


#Normalization
scaler = MinMaxScaler()
scaler.fit(df[:, 1:])
df[:, 1:] = scaler.transform(df[:, 1:])

# 10-Fold CV
##First randomly shuffle the data orderd
df = shuffle(df)


print("#############")

#then split data into 10 folds
num_folds = 10

folds = np.array_split(df, num_folds)

print(folds)

mse = []
#Iteration
for i in range (num_folds):
    xtest = folds[i][:,:7] # Set ith fold to be test
    ytest = folds[i][:,7]
    new_folds = np.row_stack(np.delete(folds,i,0))
    xtrain = new_folds[:, :7]
    ytrain = new_folds[:,7]

    # some print functions to help you debug
    print(f'Fold {i}')
    print(f'xtest shape  : {xtest.shape}')
    print(f'ytest shape  : {ytest.shape}')
    print(f'xtrain shape : {xtrain.shape}')
    print(f'ytrain shape : {ytrain.shape}\n')

    w = np.array([0, 0, 0, 0, 0, 0, 0])
    b = 0

    # learning rate
    alpha = 0.01

    # gd

    for i in range(50000):
        w = w - alpha * (1 / len(folds)) * np.dot(np.transpose(np.dot(xtrain, w) + b - ytrain), xtrain)
        b = b - alpha * (1 / len(folds)) * sum(np.dot(xtrain, w) + b - ytrain)

    print(w, b)
#    mse.append(np.square(np.subtract(xtrain,ytrain)).mean())




