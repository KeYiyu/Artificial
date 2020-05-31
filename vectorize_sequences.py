# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:41:46 2019

@author: Mr k
"""
from keras.utils.np_utils import to_categorical
import numpy as np
def vectorize_sequences(sequences,dimension=8):
    results=np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence]=1.
    return results
x=[[2,3,4,5,6],[1,2,3,4,5],[3,4,5,6,7]]
'''z=to_categorical(x)
for i in range(3):
    for j in range(5):
        print(z)'''
'''print(z)'''
z1=vectorize_sequences(x)
print(z1)