# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 08:53:53 2019

@author: Mr k
"""
from keras.models import Model
from keras import layers
from keras import Input
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import keras

base_dir='D:/cats dogs'
train_dir=os.path.join(base_dir,'train')
validation_dir=os.path.join(base_dir,'validation')
test_dir=os.path.join(base_dir,'test')
train_cats_dir=os.path.join(train_dir,'cats')
train_dogs_dir=os.path.join(train_dir,'dogs')
validation_cats_dir=os.path.join(validation_dir,'cats')
validation_dogs_dir=os.path.join(validation_dir,'dogs')
test_cats_dir=os.path.join(test_dir,'cats')
test_dogs_dir=os.path.join(test_dir,'dogs')


in_put=Input(shape=(150,150,3),name='text')
x=layers.Conv2D(32,3,activation='relu')(in_put)
x=layers.MaxPool2D(2)(x)
x=layers.Conv2D(64,3,activation='relu')(x)
x=layers.MaxPool2D(2)(x)
x=layers.Conv2D(128,3,activation='relu')(x)
x=layers.MaxPool2D(2)(x)
x=layers.Conv2D(128,3,activation='relu')(x)
x=layers.MaxPool2D(2)(x)
x=layers.Flatten()(x)
x=layers.Dropout(0.5)(x)
x=layers.Dense(512,activation='relu')(x)
out_put=layers.Dense(1,activation='sigmoid')(x)
model=Model(in_put,out_put)
model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

train_datagen=ImageDataGenerator(rescale=1./255,
                                 rotation_range=40,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(
        train_dir,
        target_size=(150,150),
        batch_size=32,
        class_mode='binary')

validation_generator=test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150,150),
        batch_size=32,
        class_mode='binary')


callbacks_list=[
        keras.callbacks.EarlyStopping(
                monitor='val_acc',
                patience=3,),
                keras.callbacks.TensorBoard(
                        log_dir='logs/cats dogs',
                        )
                ]
        
history=model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=100,
        callbacks=callbacks_list,
        validation_data=validation_generator,
        validation_steps=50)
