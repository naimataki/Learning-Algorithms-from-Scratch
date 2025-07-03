from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

#print(df[['MedInc', 'MedHouseVal']].head())

X = df[['MedInc']].iloc[:200]  # Keep 2D
y = df['MedHouseVal'].iloc[:200]  # Target 

#plt.scatter(X, y)
#plt.show()

def loss_function(m, b, X, y):
    total_error = 0
    n = len(X)
    for i in range(n):
        total_error += (y.iloc[i] - (m*X.iloc[i] + b))**2
    return total_error / float(n)

def gradient_descent(m, b, X, y, alpha):
    dm = 0
    db = 0
    n = len(X)

    for i in range(n):
        dm += - (2/n) * X.iloc[i] * (y.iloc[i] - (m*X.iloc[i] + b))
        db += - (2/n) * (y.iloc[i] - (m*X.iloc[i] + b))

    m -= alpha * dm
    b -= alpha * db

    return m, b

m = 0
b = 0
alpha = 0.01
epochs = 100

for _ in range(epochs):
    m, b = gradient_descent(m, b, X, y, alpha)

print(m, b)

x_line = np.linspace(X.min()[0], X.max()[0], 100)
y_line = [m * x + b for x in x_line]

plt.scatter(X, y, color = 'red')
plt.plot(x_line, y_line)
plt.show()


