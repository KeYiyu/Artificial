# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 09:40:22 2019

@author: Mr k
"""

import numpy as np
import string
from keras.preprocessing.text import Tokenizer

samples=['The cat sat on the mat.','The dog ate my homework.']


#单词级别

token_index={}
for sample in samples:
    for word in sample.split():
        #print(word)
        token_index[word]=len(token_index)+1
        #print(token_index[word])
#print(token_index)
max_length=10
results=np.zeros(shape=(len(samples),
                        max_length,
                        max(token_index.values())+1))
for i,sample in enumerate(samples):
    for j,word in list(enumerate(sample.split()))[:max_length]:
        index=token_index.get(word)
        results[i,j,index]=1.
        
print('pattern1:')   
print(results)



#字符级别
'''
characters=string.printable
token_index=dict(zip(characters,range(1,len(characters)+1)))
#print(token_index)
max_length=50
results=np.zeros((len(samples),max_length,max(token_index.values())+1))
for i,sample in enumerate(samples):
    for j,character in enumerate(sample):
        #print(character)
        index=token_index.get(character)
        #print(index)
        results[i,j,index]=1.
    
print('pattern2:')
print(results)
'''


#keras 实现单词级 one hot 编码
'''
tokenizer=Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)
sequences=tokenizer.texts_to_matrix(samples,mode='binary')
word_index=tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
'''

#使用散列技巧的单词级 one hot
'''
dimensionality=1000
max_length=10

results=np.zeros((len(samples),max_length,dimensionality))
for i,sample in enumerate(samples):
    for j,word in list(enumerate(sample.split()))[:max_length]:
        index=abs(hash(word))%dimensionality
        results[i,j,index]=1.
print(results)
'''