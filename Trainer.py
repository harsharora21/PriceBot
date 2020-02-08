import tensorflow as tf
import numpy as np
import math
import datamanu as dm
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras.backend as kb

#Custom cost function
def moneyStrat(corr,pre):
    money=-kb.dot(kb.transpose(corr),pre)
    return money

dataname=input("Enter Data File Name: ")
delay=int(input("Enter Delay: "))
modelname=input("Enter Model Name: ")

tsrs=np.load(dataname)
mmax=5000
mmin=-5000
tsrs=dm.normalize(tsrs,mmax,mmin)

x_data,y_data=dm.convTimeSeries(list(tsrs),delay)
x_data=np.array(x_data)
y_data=dm.convToClassification(x_data,y_data)
y_data=np.array(y_data)

testrat=0.75
spln=math.floor(testrat*y_data.size)

x_train=x_data[0:spln]
y_train=y_data[0:spln]

x_test=x_data[spln:]
y_test=y_data[spln:]

myact='selu'
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(128, activation=myact,input_shape=[delay]),
tf.keras.layers.Dense(128, activation=myact),
tf.keras.layers.Dense(128, activation=myact),
tf.keras.layers.Dense(1, activation='softsign')
])

opti=tf.keras.optimizers.RMSprop(0.001)

model.compile(optimizer=opti,
loss=moneyStrat)

model.fit(x_train, y_train, epochs=150)

model.evaluate(x_test,  y_test, verbose=2)
model.save(modelname)

'''
Sample way to load and Test Model
model = tf.keras.models.load_model('blazev2')
yp=dm.denormalize(model.predict(x_test),mmax,mmin)
y=dm.denormalize(y_test,mmax,mmin)
'''
