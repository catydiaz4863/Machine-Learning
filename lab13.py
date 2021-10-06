import numpy as np
import matplotlib.pyplot as pltp
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

df = np.loadtxt("/Users/catalinadiaz/Documents/ML_fall2021/iris.data", usecols=(0,1,2,3), delimiter=",")


#Selecting the first 100 samples
df = df[:100, :]

#Boolean Class

df = pd.get_dummies(df)



print(df)