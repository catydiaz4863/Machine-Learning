from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np

df = pd.read_fwf("/Users/catalinadiaz/Documents/ML_fall2021/auto-mpg_removed_missing_values.data", delimiter = " ", usecols=(0,1,2,3,4,5,6,7))
df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)



#Preparing datasets
X = df.values

#Normalization
scaler = MinMaxScaler()
scaler.fit(X[:, 1:])
X[:, 1:] = scaler.transform(X[:, 1:])
print(X)