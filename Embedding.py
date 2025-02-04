# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 13:19:18 2019

@author: Mr k
"""

import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Sequential


imdb_dir='D:/deep learning/aclImdb'
train_dir=os.path.join(imdb_dir,'train')

labels=[]
texts=[]

for label_type in ['neg','pos']:
    dir_name=os.path.join(train_dir,label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:]=='.txt':
            f=open(os.path.join(dir_name,fname))
            texts.append(f.read())
            f.close()
            if label_type=='neg':
                labels.append(0)
            else:
                labels.append(1)
    
maxlen=100
training_samples=200
validation_samples=1000
max_words=1000
tokenizer=Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences=tokenizer.texts_to_sequences(texts)
word_index=tokenizer.word_index
print('Found %s unique token.'% len(word_index))
data=pad_sequences(sequences,maxlen=maxlen)
labels=np.asarray(labels)
print('Shape of data tensor:',data.shape)
print('Shape of labels tensor:',labels.shape)
indices=np.arange(data.shape[0])
np.random.shuffle(indices)
data=data[indices]
labels=labels[indices]
x_train=data[:training_samples]
y_train=labels[:training_samples]
x_val=data[training_samples:training_samples+validation_samples]
y_val=labels[training_samples:training_samples+validation_samples]



glove_dir='D:/deep.learning'

embeddings_index={}
f=open(os.path.join(glove_dir,'glove.6B.100d.txt'),'r',encoding='UTF-8')
for line in f:
    values=line.split()
   # print(values)
    word=values[0]
    coefs=np.asarray(values[1:],dtype='float32')
    embeddings_index[word]=coefs
f.close()
print('Found %s word vectors.'% len(embeddings_index))


embedding_dim=100
embedding_matrix=np.zeros((max_words,embedding_dim))
for word,i in word_index.items():
    if i<max_words:
        embedding_vector=embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i]=embedding_vector
            
            
model=Sequential()
model.add(Embedding(max_words,embedding_dim,input_length=maxlen))    

    
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable=False

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history=model.fit(x_train,y_train,
                  epochs=10,
                  batch_size=32,
                  validation_data=(x_val,y_val))
model.save_weights('pre_trained_glove_model.h5')