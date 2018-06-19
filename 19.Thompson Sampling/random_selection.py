# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 10:57:50 2018

Random Selection

@author: Arnob
"""

# import lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# implementing random Selection
import random
N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward
    
# Visualising the results - histrogram
plt.hist(ads_selected)
plt.title('Histrogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()    


