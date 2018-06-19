# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 00:47:06 2018

Thompson Sampling

@author: Arnob
"""

# import lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt,log
import random

# importing dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')


# implementing ucb
N = 10000
d = 10            # number of ad
ads_selected = [] # empty vector/list it will appand ads

number_of_rewards_1 = [0] * d    # vector d eliments 
number_of_rewards_0 = [0] * d

total_reward = 0

for n in range(0, N):
    max_random = 0
    ad = 0
    for i in range(0, d):
        random_beta = random.betavariate(number_of_rewards_1[i] + 1, number_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_upper_bound = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    
    if reward == 1:
        number_of_rewards_1[ad] = number_of_rewards_1[ad] + 1
    else:
        number_of_rewards_0[ad] = number_of_rewards_0[ad] + 1
        
    total_reward = total_reward + reward
        
# Visulization of the result
plt.hist(ads_selected)      
plt.title('Histrogram Of ad selection')
plt.xlabel('ads')
plt.ylabel('Number of time each ad was selected')
plt.show()  
        