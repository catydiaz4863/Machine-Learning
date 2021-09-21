import splits as splits
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn.utils import shuffle


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




