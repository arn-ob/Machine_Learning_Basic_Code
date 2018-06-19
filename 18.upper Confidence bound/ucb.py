# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 00:24:11 2018

Upper Confidence Bound

@author: Arnob
"""


# import lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt,log


# importing dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')


# implementing ucb
N = 10000
d = 10            # number of ad
ads_selected = [] # empty vector/list it will appand ads

number_of_selections = [0] * d
sum_of_rewards = [0] * d 
total_reward = 0

for n in range(0, N):
    max_upper_bound = 0
    ad = 0
    for i in range(0, d):
        if (number_of_selections[i] > 0):
            avgerage_reward = sum_of_rewards[i] / number_of_selections[i]
            delta_i = sqrt(3/2 * log(n + 1)/ number_of_selections[i])
            upper_bound = avgerage_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    number_of_selections[ad] = number_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sum_of_rewards[ad] = sum_of_rewards[ad] + reward
    total_reward = total_reward + reward
        
# Visulization of the result
plt.hist(ads_selected)      
plt.title('Histrogram Of ad selection')
plt.xlabel('ads')
plt.ylabel('Number of time each ad was selected')
plt.show()  
        
        
