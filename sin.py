# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 09:12:43 2019

@author: Mr k
"""
import keras
from keras import models
from keras import layers
from keras import optimizers
import math
#import numpy as np
#x=np.array()
x=[]
y=[]
for i in range(1000):
    x.append(i)
    y.append(0.5*math.sin(3+2*i))
x1=x[0:600]
x2=x[600:800]
x3=x[800:1000]
y1=y[0:600]
y2=y[600:800]
y3=y[800:1000]
model=models.Sequential()
model.add(layers.Dense(4,activation='relu',input_shape=(1,)))
model.add(layers.Dense(4,activation='relu'))
model.add(layers.Dense(4,activation='relu'))
model.add(layers.Dense(1,activation='tanh'))

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='mse',
              metrics=['mae'])
callbacks_list=[
        keras.callbacks.EarlyStopping(
                monitor='val_acc',
                patience=3,),
                keras.callbacks.TensorBoard(
                        log_dir='logs/sin'
                        )
                ]
his=model.fit(x1,
              y1,
              epochs=10,
              batch_size=100,
              callbacks=callbacks_list,
              validation_data=(x[600:800],y[600:800]),
              )
his_dit=his.history
#acc=his_dit['acc']
loss=his_dit['loss']
val_loss=his_dit['val_loss']
print(val_loss,loss)

test1,test2=model.evaluate(x[800:1000],y[800:1000])
print(test1,test2)