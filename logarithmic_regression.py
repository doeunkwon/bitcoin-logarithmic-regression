import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

csv_name = 'btc-market-price.csv'
date = 'timestamp'
value = 'market-price'

dataframe = pd.read_csv(csv_name)
dataframe = dataframe[dataframe[value] > 0]
dataframe[date] = pd.to_datetime(dataframe[date])

def log_reg(days, a, b):
    return a * np.log(days) + b

x_data = np.array([x + 1 for x in range(len(dataframe))])
y_data = np.log(dataframe[value])
opt, cov = curve_fit(log_reg, x_data, y_data)
fitted_y_data = log_reg(x_data, opt[0], opt[1])

plt.style.use("fivethirtyeight")

plt.semilogy(dataframe[date], dataframe[value], linewidth=0.7)

num = -2
for col in ['tab:purple','tab:blue','tab:green','y','tab:orange','tab:red']:
    plt.plot(dataframe[date], np.exp(fitted_y_data + num), color=col, linewidth=0.7)
    num += 0.87

plt.ylim(bottom = 1)
plt.show()
