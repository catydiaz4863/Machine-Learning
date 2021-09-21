import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as stats


df = np.loadtxt("/Users/catalinadiaz/Documents/ML_fall2021/auto-mpg_removed_missing_values.data", usecols=(0,1,2,3,4,5,6,7))

print(df)
print("####################")
print(df[:, 0])
