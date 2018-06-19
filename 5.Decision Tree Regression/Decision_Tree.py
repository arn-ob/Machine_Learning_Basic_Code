# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 14:50:51 2018

@author: Arnob
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values  

# Fitting the Decision Tree Regression to the db
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)
 

# Predicting  a new result
y_pred = regressor.predict(6.5)

# Visualisin for the regressor for higher and smoother curve
# To see the accure prediction
X_grid = np.arange(min(X), max(X), 0.10)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X,y, color= 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Decession Tree Lookup')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()



# Visualising the SVR result to view the basic
plt.scatter(X,y, color= 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Decession Tree Lookup')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()