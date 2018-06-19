# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 13:44:17 2018

Apriori

@author: Arnob
"""


# import lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

# traning  Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
# at min_support = 3(purches time per week)*7(per week)/7500 (Total purches) [3*7/7500]
# at min_confidence = 20%

# visualising the results
results = list(rules)





   