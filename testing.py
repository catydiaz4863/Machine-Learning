
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


data = np.loadtxt("/Users/catalinadiaz/Documents/ML_fall2021/auto-mpg_removed_missing_values.data", usecols=(0,1,2,3,4,5,6,7))

scaler = MinMaxScaler()
scaler.fit(data)
data = scaler.transform(data)

k = 10
np.random.shuffle(data)
splitData = np.array_split(data, k)
print(data)
mseArray = []
# for split in splitData[1:]:
#     print(len(split))

for i in range(k):
    print(i)
    test = splitData[i]
    train = np.concatenate(splitData[:i] + splitData[i+1:])

    trainY = train[:, 0]  # mpg value
    trainX = train[:, 1:]  # rest of the values
    trainN = len(train)

    testY = test[:, 0]  # mpg value
    testX = test[:, 1:]  # rest of the values
    testN = len(test)

    assert len(trainX) == len(trainY), 'Length of X and Y different'
    # initialization
    w = np.zeros(len(trainX[0]))
    b = 0
    # learning rate
    alpha = 0.01

#    print(f'Fold {i}')
#    print(f'xtest shape  : {testX.shape}')
#    print(f'ytest shape  : {testY.shape}')
#    print(f'xtrain shape : {trainX.shape}')
#    print(f'ytrain shape : {trainY.shape}')
#    print(f'train : {trainN}\n')
    # Gradient Descent
    for i in range(10000):
        w = w - alpha * (1 / trainN) * np.dot(np.transpose(np.dot(trainX, w) + b - trainY), trainX)
        b = b - alpha * (1 / trainN) * sum(np.dot(trainX, w) + b - trainY)

    print(w, b)

    predY = np.dot(testX, w) + b
    mse = sum(np.square(predY - testY)) / testN
    print(mse)
    mseArray.append(mse)

print('Average MSE:')
print(sum(mseArray)/len(mseArray))